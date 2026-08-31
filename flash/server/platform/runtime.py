"""Background lifecycle, restart recovery, and worker-artifact helpers.

This module must remain free of fastapi and optional server extras for app import time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading

from flash.adapters.artifacts import attempt_scoped_artifact_name
from flash.core.spec import JobSpec
from flash.runner.lifecycle.state import adapter_prefix, runs_file_path
from flash.runner.lifecycle.status import get_status
from flash.server.platform import db

_log = logging.getLogger("flash.server")

# Run states that are still in flight and must be recovered after a control-plane restart.
_RECOVERABLE = {"queued", "provisioning", "running"}


async def _reconcile_cost_loop() -> None:
    """Background loop: periodically pull realized provider cost (COGS) for finished runs and
    report it to the freesolo backend for estimator accuracy. The provider billing calls are
    blocking urllib, so each sweep is offloaded to a thread; failures are swallowed and retried
    next cycle. Off entirely when FREESOLO_INTERNAL_KEY is unset (see reconcile_enabled)."""
    from flash.server.domain.ops.reconcile import reconcile_once

    interval = 3600.0  # COGS reconcile sweep interval (fixed; flash is fully managed)
    while True:
        await asyncio.sleep(interval)
        # Handle cancellation EXPLICITLY (re-raise it) and swallow only real Exceptions, exactly
        # like the sibling reaper loops in app.py (_reap_idle_endpoints_loop /
        # _sweep_orphan_instances_loop). On the supported Pythons (>=3.11) asyncio.CancelledError
        # already derives from BaseException, so the old `contextlib.suppress(Exception)` did not
        # swallow a shutdown cancel arriving during the blocking sweep — but being explicit makes the
        # cancel path obvious and uniform, and logs a failed sweep instead of silently dropping it.
        try:
            reported = await asyncio.to_thread(reconcile_once)
            if reported:
                _log.info("reconciled realized cost for %d run(s)", reported)
        except asyncio.CancelledError:
            raise  # shutdown: let the lifespan's task.cancel() propagate, don't swallow it
        except Exception:
            _log.debug("realized-cost reconcile sweep failed; retrying next cycle", exc_info=True)


async def _repo_cleanup_loop() -> None:
    """Sweep aged undeployed HF prefixes on startup, then daily.

    The 7-day age and daily cadence are fixed in ``flash.server.domain.repo_cleanup``. Each threaded sweep
    fails closed when the live serving set is unconfirmed and requires operator ``HF_TOKEN``.
    """
    from flash.server.domain.ops.repo_cleanup import CleanupAborted, run_scheduled_cleanup

    interval = (
        24.0 * 3600.0
    )  # daily sweep (fixed); the 7-day age makes the exact cadence non-critical
    # Sweep IMMEDIATELY on startup, THEN sleep between subsequent sweeps (sleep is at the END of the
    # loop). Sleeping first would let a control plane restarted — or crash-looping — more often than
    # the interval never reclaim anything. The startup sweep is still off the critical path — this is a
    # background task and the sweep itself runs in a worker thread (see app.lifespan).
    while True:
        # The blocking sweep runs in a worker thread that task.cancel() can't interrupt; a stop Event
        # lets the lifespan signal it to halt BETWEEN deletes at shutdown, so a large in-flight sweep
        # can't keep deleting after the server was told to stop (see _reconcile_cost_loop).
        stop = threading.Event()
        try:
            deleted = await asyncio.to_thread(run_scheduled_cleanup, should_stop=stop.is_set)
            if deleted:
                _log.info("repo GC: deleted %d aged undeployed run prefix(es)", deleted)
        except asyncio.CancelledError:
            stop.set()  # signal the worker thread to stop deleting between targets
            raise  # shutdown: let the lifespan's task.cancel() propagate, don't swallow it
        except CleanupAborted as exc:
            # Live set unconfirmed -> the sweep deleted NOTHING by design. Expected during a serving/
            # registry blip; not an error. Retry next cycle.
            _log.warning(
                "repo GC sweep skipped (live set unconfirmed); retrying next cycle: %s", exc
            )
        except Exception:
            _log.debug("repo GC sweep failed; retrying next cycle", exc_info=True)
        # Sleep AFTER the sweep, so the first sweep runs at startup rather than a full interval later.
        await asyncio.sleep(interval)


async def _charge_retry_startup() -> None:
    """Run one completion-charge recovery sweep outside the startup critical path.

    It must run promptly because ``recover_runs`` excludes terminal ``done``, but blocking urllib
    charges must not delay lifespan startup. The background thread is best-effort and cancellable.
    """
    from flash.server.billing.retry import retry_completion_charges_once

    stop = threading.Event()
    try:
        recovered = await asyncio.to_thread(retry_completion_charges_once, stop.is_set)
        if recovered:
            _log.info("recovered %d pending completion charge(s) at startup", recovered)
    except asyncio.CancelledError:
        # task.cancel() only cancels the await, not the worker thread; signal it to stop between
        # runs so a backlog of slow charges can't keep the thread alive past shutdown.
        stop.set()
        raise  # shutdown during the startup sweep: let the lifespan's task.cancel() propagate
    except Exception:
        _log.debug(
            "startup completion-charge sweep failed; periodic loop will retry", exc_info=True
        )


async def _charge_retry_loop() -> None:
    """Retry incomplete charges for finished runs on a bounded interval.

    Charges are idempotent by runId and run in a thread because urllib blocks. Disabled without
    ``FREESOLO_INTERNAL_KEY``; failures retry on the next interval.
    """
    from flash.server.billing.retry import retry_completion_charges_once

    interval = 300.0  # retry pending/failed customer charges every 5 min
    while True:
        await asyncio.sleep(interval)
        # Mirror the sibling reaper/reconcile loops: re-raise CancelledError so the lifespan's
        # task.cancel() stops it at shutdown, and swallow only real Exceptions to keep sweeping.
        stop = threading.Event()
        try:
            charged = await asyncio.to_thread(retry_completion_charges_once, stop.is_set)
            if charged:
                _log.info("recovered %d pending completion charge(s) on retry", charged)
        except asyncio.CancelledError:
            stop.set()  # signal the worker thread to stop between runs (see _charge_retry_startup)
            raise  # shutdown: let the lifespan's task.cancel() propagate, don't swallow it
        except Exception:
            _log.debug("completion-charge retry sweep failed; retrying next cycle", exc_info=True)


def _append_run_log(run_id: str, message: str) -> None:
    """Append a timestamped note to a run's log so it surfaces in `flash runs log`."""
    import time

    with open(runs_file_path(run_id, ".log"), "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")


# a handle-less recovery resubmit remains deferred until strict absence is proven. observation is bounded
# by the run wall deadline; after it passes, terminal persistence keeps retrying until durably confirmed.
_DEFERRED_RECOVERY_RETRY_S = 120.0


def _confirm_run_clear(spec) -> bool:
    """Force-reap this run's label and confirm no instance remains.

    ``gc`` alone cannot prove Vast teardown. Providers with ``run_instances_remaining`` must return
    ``[]``; a live instance or enumeration failure blocks resubmit. A provider recorded in
    ``submitted_instance_providers`` also blocks when it is now unconfigurable, because a lost
    non-idempotent create such as Vast ``PUT /asks`` may still be billing without a handle.
    """
    from flash.providers.core.registry import INSTANCE_PROVIDERS, configured_providers, get_provider

    try:
        recorded_raw = getattr(get_status(spec.run_id), "submitted_instance_providers", None)
    except Exception:
        # Can't read the durable record of providers available at submit -> can't rule out a recorded
        # Vast phantom that this restart can no longer enumerate -> fail CLOSED.
        return False
    recorded = {str(name) for name in recorded_raw} if recorded_raw is not None else None
    configured = {getattr(p, "name", None): p for p in configured_providers()}
    clear = True
    for name, prov in configured.items():
        if name not in INSTANCE_PROVIDERS:
            continue
        if recorded is not None and name not in recorded:
            continue
        with contextlib.suppress(Exception):
            prov.gc(spec)
        check = getattr(prov, "run_instances_remaining", None)
        if check is None:
            continue  # provider exposes no confirmation -> preserve best-effort resubmit
        try:
            if check(spec.run_id):
                clear = False  # an instance for this run is still present
        except Exception:
            clear = False  # couldn't list -> can't prove clear -> don't race
    # An instance provider that WAS available at submit (so it could have taken the lost create) but
    # is NOT configurable now and owns the standing-instance capability can't be enumerated -> can't
    # prove clear -> fail closed. Already-configured providers were handled above. CRITICAL: do NOT
    # wrap this in a broad ``suppress(Exception)`` — an error loading the status, resolving a
    # recorded provider, or reading the capability would otherwise be swallowed and leave ``clear``
    # True, defeating the fail-closed intent. Every failure inspecting the recorded set must instead
    # make the guard CONSERVATIVE (block/defer the resubmit).
    for name in recorded or []:
        if name in configured or name not in INSTANCE_PROVIDERS:
            continue
        try:
            has_capability = (
                getattr(get_provider(name), "run_instances_remaining", None) is not None
            )
        except Exception:
            # Can't even resolve a RECORDED instance provider -> assume it could own a phantom -> fail
            # closed rather than declare clear.
            has_capability = True
        if has_capability:
            clear = False
    return clear


def _recovery_block_reason(spec) -> str | None:
    from flash.runner.lifecycle.deadlines import _spec_with_remaining_wall

    try:
        _spec_with_remaining_wall(spec, require_provider_minimum=True)
    except RuntimeError as exc:
        return str(exc)
    return None


def _recovery_wall_deadline_is_open(spec) -> bool:
    from flash.runner.lifecycle.deadlines import _remaining_run_wall_seconds

    try:
        return _remaining_run_wall_seconds(spec.run_id) > 0
    except Exception:
        return False


def _fail_blocked_recovery(
    spec,
    reason: str,
    *,
    expected_remote: dict | None = None,
) -> bool:
    from flash.runner.accounting.reconciliation import _compare_and_fail_remote
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.supervise.lifecycle import _adopt_completed_attempt, _CompletedAttemptPending

    status = get_status(spec.run_id)
    if status.remote is None:
        try:
            deadline_at = _load_run_deadline_at(spec.run_id)
        except RuntimeError:
            deadline_at = float(status.created_at) + float(spec.gpu.max_wall_seconds)
        try:
            metrics = _handleless_completed_metrics(spec, status, deadline_at)
        except _CompletedAttemptPending:
            return False
        if metrics is not None:
            applied = _adopt_completed_attempt(
                spec.run_id,
                spec,
                None,
                metrics,
                log=None,
            )
            if applied:
                with contextlib.suppress(Exception):
                    _append_run_log(
                        spec.run_id,
                        "adopted a completed attempt before failing blocked recovery",
                    )
            return applied

    applied = _compare_and_fail_remote(spec.run_id, expected_remote, reason)
    if applied:
        with contextlib.suppress(Exception):
            _append_run_log(spec.run_id, reason)
    return applied


def _start_handleless_resubmit(spec, expected_state: str) -> bool | None:
    """Claim one provider-clear handleless launch without advancing an active claim."""
    from flash.runner.lifecycle.attempts import (
        _verified_opd_retry_state,
        active_launch_claim_from_raw,
        claim_is_live,
        reserve_handleless_recovery_launch,
    )
    from flash.runner.lifecycle.status import (
        _load_status_json,
        decode_next_attempt,
        source_snapshot_from_status,
    )
    from flash.runner.supervise.lifecycle import _run_job_background

    try:
        source_snapshot_from_status(get_status(spec.run_id), required=True)
    except Exception as exc:
        _fail_blocked_recovery(spec, str(exc), expected_remote=None)
        return False
    reason = _recovery_block_reason(spec)
    if reason is not None:
        if _recovery_wall_deadline_is_open(spec):
            return False
        _fail_blocked_recovery(spec, reason, expected_remote=None)
        return False
    raw = _load_status_json(spec.run_id)
    stale_claim = active_launch_claim_from_raw(raw)
    if stale_claim is not None and claim_is_live(spec.run_id, stale_claim):
        return None
    revision = world_size = None
    verified_attempt = None
    if spec.algorithm == "opd":
        # reclaiming a stale claim is not exempt. the lost worker can cross `optimizer.step()` and
        # upload its mutation marker before provider cleanup finds and terminates it, so the claim's
        # pre-launch resume evidence is already stale. reverify every attempt the counter names and
        # carry that revision instead of the one the claim pinned.
        try:
            verified_attempt, revision, world_size = _verified_opd_retry_state(spec.run_id)
            if verified_attempt != decode_next_attempt(raw):
                raise RuntimeError("opd recovery attempt identity changed")
        except Exception as exc:
            _fail_blocked_recovery(spec, str(exc), expected_remote=None)
            return False
    reservation = reserve_handleless_recovery_launch(
        spec.run_id,
        expected_state=expected_state,
        provider_clear_confirmed=True,
        expected_stale_claim=stale_claim,
        expected_next_attempt=verified_attempt,
        resume_revision=revision,
        resume_world_size=world_size,
    )
    if reservation.active:
        return None
    if reservation.claim is None:
        if reservation.retry_plan is not None and not reservation.retry_plan.retry:
            _fail_blocked_recovery(
                spec,
                "retry policy rejected handleless replacement",
                expected_remote=None,
            )
            return False
        return None
    with contextlib.suppress(Exception):
        _append_run_log(
            spec.run_id,
            "control plane restarted without a durable handle; resubmitting",
        )
    try:
        threading.Thread(
            target=_run_job_background,
            args=(spec, None, reservation.claim),
            daemon=True,
        ).start()
    except Exception:
        # nothing will ever run `_run_job_background`, so its `finally` cannot consume the claim.
        # the caller retries, and an unconsumed claim reads as live to each later pass, which then
        # defers until the wall deadline instead of launching the recovered run.
        from flash.runner.supervise.lifecycle import _consume_reserved_claim

        _consume_reserved_claim(spec.run_id, reservation.claim)
        raise
    return True


def _handleless_completed_metrics(spec, status, deadline_at: float) -> dict | None:
    from flash.runner.lifecycle.attempts import latest_reserved_attempt
    from flash.runner.supervise.lifecycle import _completed_attempt_metrics

    attempt = latest_reserved_attempt(spec.run_id)
    if attempt is None:
        return None
    providers = {
        str(name)
        for name in (getattr(status, "submitted_instance_providers", None) or [])
        if name in {"vast", "lambda"}
    }
    for provider in sorted(providers):
        metrics = _completed_attempt_metrics(
            spec,
            provider=provider,
            attempt=attempt,
            launch_floor=float(status.created_at),
            deadline_at=deadline_at,
        )
        if metrics is not None:
            return metrics
    return None


def _deferred_resubmit_loop(spec) -> None:
    """Reconcile a handle-less maybe-live instance through the run wall deadline."""
    import time

    from flash.runner.accounting.reconciliation import _compare_and_fail_remote
    from flash.runner.lifecycle.deadlines import _load_run_deadline_at
    from flash.runner.lifecycle.state import TERMINAL_STATES
    from flash.runner.supervise.lifecycle import _adopt_completed_attempt

    while True:
        try:
            status = get_status(spec.run_id)
        except Exception:
            time.sleep(_DEFERRED_RECOVERY_RETRY_S)
            continue
        if status.state in TERMINAL_STATES or status.remote is not None:
            return
        try:
            deadline_at = _load_run_deadline_at(spec.run_id)
        except Exception as exc:
            try:
                if _compare_and_fail_remote(spec.run_id, None, str(exc)):
                    return
            except Exception:
                time.sleep(_DEFERRED_RECOVERY_RETRY_S)
                continue
            return
        if time.time() >= deadline_at:
            try:
                metrics = _handleless_completed_metrics(spec, status, deadline_at)
            except Exception:
                time.sleep(_DEFERRED_RECOVERY_RETRY_S)
                continue
            if metrics is not None:
                try:
                    _adopt_completed_attempt(
                        spec.run_id,
                        spec,
                        None,
                        metrics,
                        log=None,
                    )
                except Exception:
                    time.sleep(_DEFERRED_RECOVERY_RETRY_S)
                    continue
                return
            reason = "run wall deadline exhausted while provider teardown remained unconfirmed"
            try:
                if _compare_and_fail_remote(spec.run_id, None, reason):
                    with contextlib.suppress(Exception):
                        _append_run_log(spec.run_id, reason)
                    return
            except Exception:
                time.sleep(_DEFERRED_RECOVERY_RETRY_S)
                continue
            return
        try:
            clear = _confirm_run_clear(spec)
        except Exception:
            clear = False
        if clear:
            try:
                started = _start_handleless_resubmit(spec, status.state)
                if started is None:
                    time.sleep(_DEFERRED_RECOVERY_RETRY_S)
                    continue
            except Exception:
                time.sleep(_DEFERRED_RECOVERY_RETRY_S)
                continue
            if started:
                return
            try:
                if get_status(spec.run_id).state in TERMINAL_STATES:
                    return
            except Exception:
                pass
            delay = min(_DEFERRED_RECOVERY_RETRY_S, max(0.0, deadline_at - time.time()))
            if delay > 0:
                time.sleep(delay)
            continue
        delay = min(_DEFERRED_RECOVERY_RETRY_S, max(0.0, deadline_at - time.time()))
        if delay > 0:
            time.sleep(delay)


def _latest_worker_artifact_name(repo: str, prefix: str, phase: str, kind: str) -> str:
    """Return the newest worker artifact under a prefix.

    Attempt-scoped console and error artifacts use the highest uploaded attempt, which can still be
    older than the live attempt while its replacement has not uploaded yet. Console falls back to its
    canonical final name and errors to attempt 0 when listing fails; derive scoped names from the
    writer's definition to prevent silent reader/writer drift.
    """
    import re

    default = (
        f"{kind}_{phase}.txt" if kind == "console" else attempt_scoped_artifact_name(kind, phase, 0)
    )
    try:
        from huggingface_hub import HfApi

        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            repo_id=repo, repo_type="dataset"
        )
    except Exception:
        return default
    pat = re.compile(rf"^{re.escape(prefix)}/{kind}_{re.escape(phase)}(?:_attempt(\d+))?\.txt$")
    best: int | None = None
    best_name: str | None = None
    for f in files:
        m = pat.match(f)
        if not m:
            continue
        attempt = int(m.group(1)) if m.group(1) is not None else -1
        if best is None or attempt > best:
            best = attempt
            best_name = os.path.basename(f)
    return default if best_name is None else best_name


def _latest_error_artifact_name(repo: str, prefix: str, phase: str) -> str:
    """Newest worker error file under prefix."""
    return _latest_worker_artifact_name(repo, prefix, phase, "error")


def _attempt_of(name: str) -> int | None:
    """The attempt an artifact filename is scoped to."""
    import re

    m = re.search(r"_attempt(\d+)\.txt$", name)
    return int(m.group(1)) if m else None


def _ray_log_name_for_attempt(phase: str, error_name: str) -> str | None:
    """Return ray logs for the same attempt as ``error_name`` when knowable.

    Ray logs exist only for ray failures, so choosing the newest independently can pair an old raylet
    failure with a newer unrelated traceback. Build the name through the writer's shared definition
    to prevent format drift.
    """
    attempt = _attempt_of(error_name)
    if attempt is None:
        return None
    return attempt_scoped_artifact_name("raylogs", phase, attempt)


def _worker_artifacts(spec) -> dict[str, str]:
    """Fetch train-subprocess stdout and traceback from the private HF artifact repo.

    The control-plane log contains only orchestration and a truncated terminal tail. Use operator
    ``HF_TOKEN`` so ``flash runs log`` can read private worker artifacts; missing data returns ``{}``.
    """
    repo = getattr(getattr(spec, "train", None), "hf_repo", None)
    if not repo:
        return {}
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return {}
    prefix = adapter_prefix(spec)
    out: dict[str, str] = {}
    error_name = _latest_error_artifact_name(repo, prefix, spec.phase)
    # both console names, never a choice between them. the periodic snapshot and the terminal tail
    # are written to DIFFERENT destinations on purpose (a late periodic upload must not clobber the
    # terminal one), and each can be the only copy of the evidence: the scoped snapshot is the newest
    # when the worker was killed before its terminal upload ran, while the canonical name holds the
    # tail that contains the crash itself. `_latest_worker_artifact_name` scores the canonical name
    # -1, so preferring it by rank would surface a stale attempt on a retry -- and preferring the
    # scoped one alone is what hid the traceback. fetching both leaves the choice to the reader.
    console_names = dict.fromkeys(
        (
            _latest_worker_artifact_name(repo, prefix, spec.phase, "console"),
            f"console_{spec.phase}.txt",
        )
    )
    # ray's own session logs. the traceback beside them records only the downstream symptom of a
    # raylet death ("Failed to register worker to Raylet: ... End of file"), so without this the
    # collector that exists to disambiguate it uploads a diagnosis nothing ever reads back
    # (VERL-115). pinned to the TRACEBACK's attempt rather than resolved independently: they are
    # uploaded only when ray failed, so the newest pair can straddle two attempts. absent for every
    # non-ray failure, and the loop below skips it.
    ray_name = _ray_log_name_for_attempt(spec.phase, error_name)
    for name in (
        *console_names,
        error_name,
        *([ray_name] if ray_name else []),
    ):
        try:
            path = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=f"{prefix}/{name}",
                token=os.environ.get("HF_TOKEN"),
                # the worker appends to console/error files across the run, so a cached copy goes
                # stale; force a fresh pull (matches other HF artifact readers, e.g.
                # flash/providers/artifacts/hf.py:make_hf_text_reader).
                force_download=True,
            )
            # errors="replace": worker stdout can carry non-UTF-8 bytes (tracebacks, progress bars);
            # decode leniently so a single bad byte never drops the whole log on UnicodeDecodeError.
            with open(path, encoding="utf-8", errors="replace") as f:
                out[name] = f.read()
        except Exception:
            continue  # file not uploaded yet / not produced for this phase
    return out


def _teardown_unrecoverable_remote(status) -> None:
    """Stop the billable worker of a handle-backed run this build cannot activate.

    The run is already marked failed by the caller; what remains is the rented resource. Teardown
    runs off the persisted handle alone, so it works for exactly the spec the parse rejected.
    Best-effort and fully suppressed: recovery must finish for every other run regardless.
    """
    from flash.runner.accounting.reconciliation import _record_cleanup_remote
    from flash.runner.supervise.lifecycle import _strict_teardown_handle

    remote = dict(status.remote or {})
    if not remote:
        return
    try:
        deleted = _strict_teardown_handle(remote, status.run_id)
    except Exception:
        # unconfirmed deletion: leave it recorded for the cleanup drain rather than dropping it.
        deleted = False
    if not deleted:
        with contextlib.suppress(Exception):
            _record_cleanup_remote(status.run_id, remote)


def recover_runs() -> None:
    """Recover every in-flight run after a restart so a redeploy never loses a training session:
    re-attach to ``running`` jobs, and resubmit ``queued``/``provisioning`` runs that never reached
    a worker."""
    active: set[str] = set()
    # Every run this plane knows about (whatever its state). This is the multi-plane safety scope for
    # the orphan sweep below — WITHOUT it, a restarting plane sharing a Vast/Lambda account sees the
    # OTHER plane's `flash-` instances as "not active locally" and the providers' unscoped
    # known_labels=None path would DESTROY them (the same guard app.py's periodic sweep already
    # passes). `active` (handle-backed/resuming only) is the narrower must-keep set; `known` is the
    # broad don't-touch-other-planes set. Both belong on the sweep.
    known: set[str] = set()
    # Deferred until after the orphan sweep so a half-rented instance from a crashed pre-handle
    # attempt is reaped without racing the resubmit's fresh allocation.
    resubmit: list[tuple[JobSpec, str]] = []

    _classify_recoverable_runs(active, known, resubmit)
    _sweep_provider_orphans(active, known)
    _resubmit_recovered_runs(resubmit)


def _drain_cleanup_remotes_bg(run_id: str) -> None:
    from flash.runner.accounting.reconciliation import _drain_cleanup_remotes

    with contextlib.suppress(Exception):
        _drain_cleanup_remotes(run_id)


def _classify_recoverable_runs(
    active: set[str], known: set[str], resubmit: list[tuple[JobSpec, str]]
) -> None:
    """Sort every persisted run into reattach, resubmit, or terminally-failed.

    Mutates the three collections in place: ``known`` gains every run id (the multi-plane
    don't-touch scope), ``active`` only the handle-backed ones the sweep must keep, and ``resubmit``
    the handle-less ones, deferred so the orphan sweep runs before their fresh allocation.
    """
    from flash.runner.lifecycle.preparation import _mark_warmstart_source
    from flash.runner.lifecycle.status import (
        _update,
        effective_spec_from_status,
        get_status,
        reallocation_spec_from_status,
    )
    from flash.runner.supervise.attach import attach_run
    from flash.runner.supervise.recovery import _gc_run_endpoints

    for row in db.all_runs():
        known.add(row["run_id"])
        try:
            status = get_status(row["run_id"])
        except FileNotFoundError:
            continue
        # drain cleanup remotes in the background. provider outages can block each teardown through
        # retry/backoff; serial startup cleanup can exceed HEALTHCHECK grace and create a restart loop.
        # recovery below does not depend on completion.
        threading.Thread(
            target=_drain_cleanup_remotes_bg, args=(status.run_id,), daemon=True
        ).start()
        if status.state not in _RECOVERABLE:
            continue
        if status.remote is None and status.cleanup_confirmed_remote is not None:
            active.add(status.run_id)
            threading.Thread(target=lambda rid=row["run_id"]: attach_run(rid), daemon=True).start()
            continue
        if status.remote:
            # A spec this build can no longer parse cannot be reattached either: attach_run parses
            # it before its own except/finally handlers exist, so the daemon thread would die at
            # that line with the run still non-terminal and its worker still billing. That is a real
            # upgrade path -- a run accepted under an algorithm this build has since dropped -- so
            # decide it HERE, where the handle is still in hand, rather than dispatching a thread
            # that can only crash. Same disposition as the handle-less branch below: fail it
            # visibly, then tear the worker down. `_strict_teardown_handle` needs only the persisted
            # handle and the run id, never a parsed spec, so removal does not depend on the parse
            # that just failed.
            try:
                effective_spec_from_status(status)
            except Exception as exc:
                _log.warning(
                    "marking run %s failed: persisted spec could not be activated for reattach",
                    status.run_id,
                    exc_info=True,
                )
                detail = f"unrecoverable: persisted spec cannot be activated: {exc}"
                with contextlib.suppress(Exception):
                    _update(status.run_id, "failed", error=detail)
                with contextlib.suppress(Exception):
                    _append_run_log(status.run_id, detail)
                _teardown_unrecoverable_remote(status)
                # If that teardown could not confirm deletion it records the handle for the cleanup
                # drain -- but this run's drain was dispatched above and has already taken its
                # snapshot, of a list that did not yet contain this record.
                # `_drain_cleanup_remotes` returns early on an empty snapshot, so nothing would
                # retry it before the next restart. Dispatch a second drain now that the record
                # exists. A double teardown of one handle is safe: `_strict_teardown_handle` is
                # idempotent and the record is removed by compare-and-remove only on a confirmed
                # delete.
                threading.Thread(
                    target=_drain_cleanup_remotes_bg, args=(status.run_id,), daemon=True
                ).start()
                continue
            # Only handle-backed runs are kept by the sweep; a handle-less run is being
            # resubmitted, so its stale half-rented instance (if any) must NOT be shielded.
            active.add(status.run_id)
            threading.Thread(target=lambda rid=row["run_id"]: attach_run(rid), daemon=True).start()
        else:
            # No durable handle exists, but a worker may still exist if a non-idempotent create was
            # accepted while its response or handle was lost. A spec that won't parse can never be
            # resubmitted -> mark it terminally failed
            # (operator-visible, dropped from _RECOVERABLE so it isn't re-skipped every restart);
            # otherwise GC any half-made endpoint and resubmit from scratch.
            try:
                spec = JobSpec.from_dict(status.spec)
                effective_spec_from_status(status)
            except Exception as exc:
                _log.warning(
                    "marking run %s failed: persisted spec could not be activated",
                    status.run_id,
                    exc_info=True,
                )
                detail = f"unrecoverable: persisted spec cannot be activated: {exc}"
                with contextlib.suppress(Exception):
                    _update(status.run_id, "failed", error=detail)
                with contextlib.suppress(Exception):
                    _append_run_log(status.run_id, detail)
                # The aborted attempt may STILL have registered its uniquely-named RunPod
                # endpoint before crashing (the exact leak the good-spec branch's
                # `_gc_run_endpoints` guards against). The `sweep_orphans` dispatch below is a
                # no-op for RunPod, and the periodic idle reaper would only reclaim this after its
                # 15-min idle grace — so tear it down by name HERE for immediate cleanup.
                # `_gc_run_endpoints` needs a parsed `JobSpec`, which we don't have; but the
                # endpoint name is derived deterministically from the run id + GPU class
                # (`endpoint_name(gpu, _run_suffix(run_id))`), both readable from the RAW
                # persisted status without parsing the spec. Terminate by that reconstructed
                # name. Best-effort/suppressed so it can never re-abort recovery; then continue.
                from flash.providers.runpod.execution.provider import terminate_persisted_endpoints

                terminate_persisted_endpoints(status.spec, status.run_id)
                continue
            # reap run-scoped resources from the parseable public spec before touching the private
            # warm-start snapshot. source drift or snapshot tampering must not strand an endpoint.
            with contextlib.suppress(Exception):
                _gc_run_endpoints(spec)
            try:
                spec = reallocation_spec_from_status(status, verify_source=True)
                # refresh the warm-start marker so recovery keeps an aged source protected.
                _mark_warmstart_source(spec, spec.run_id)
            except Exception as exc:
                _log.warning(
                    "marking run %s failed: warm-start ref could not be resolved",
                    status.run_id,
                    exc_info=True,
                )
                detail = f"unrecoverable: train.init_from_adapter could not be resolved: {exc}"
                with contextlib.suppress(Exception):
                    _update(status.run_id, "failed", error=detail)
                with contextlib.suppress(Exception):
                    _append_run_log(status.run_id, detail)
                continue
            resubmit.append((spec, status.state))


def _sweep_provider_orphans(active: set[str], known: set[str]) -> None:
    """Reap orphaned per-run provider resources; each provider sweeps its own."""
    from flash.providers.core.registry import configured_providers

    for prov in configured_providers():
        with contextlib.suppress(Exception):
            prov.sweep_orphans(active_labels=active, known_labels=known)


def _resubmit_recovered_runs(resubmit: list[tuple[JobSpec, str]]) -> None:
    """Relaunch the handle-less runs, deferring any whose teardown could not be confirmed."""
    from flash.runner.lifecycle.status import get_status

    for spec, prior_state in resubmit:
        reason = _recovery_block_reason(spec)
        if reason is not None:
            applied = False
            if not _recovery_wall_deadline_is_open(spec):
                try:
                    applied = _fail_blocked_recovery(spec, reason)
                except Exception:
                    applied = False
            if not applied:
                try:
                    current = get_status(spec.run_id)
                except Exception:
                    current = None
                if current is None or (current.state in _RECOVERABLE and current.remote is None):
                    threading.Thread(
                        target=_deferred_resubmit_loop,
                        args=(spec,),
                        daemon=True,
                    ).start()
            continue
        _log.info("resubmitting run %s after control-plane restart", spec.run_id)
        # MtzrJ: a handle-less run hit the submit->provisioning window, so a NON-IDEMPOTENT instance
        # create (Vast's PUT /asks) may have been accepted while the response/handle was lost — a
        # phantom contract that bills and, worse, writes this run's HF artifacts. The batch
        # sweep_orphans above reaps it only if it was VISIBLE then; _confirm_run_clear force-reaps this
        # run's label across the instance providers RIGHT BEFORE relaunching and verifies nothing for the
        # run remains, so a phantom that surfaced in the sweep->resubmit gap (Vast's instance list is
        # eventually consistent) can't get a SECOND worker writing the same run-scoped artifacts.
        # A run still `queued` never reached that window: the runner enqueues as `queued` and
        # lifecycle's `_update(..., "provisioning")` fires BEFORE any `provider.submit_attempt`, and
        # nothing regresses `provisioning`->`queued`, so a `queued` run made no create and can't
        # have left a phantom. Skip
        # the guard for it — else a purely-queued run whose VAST_API_KEY was dropped after submit would
        # fail closed in _confirm_run_clear (unenumerable recorded Vast) and defer forever. The guard
        # still runs for `provisioning`/`running`, the states that could have attempted a create.
        if prior_state == "queued" or _confirm_run_clear(spec):
            if _start_handleless_resubmit(spec, prior_state):
                continue
            try:
                current = get_status(spec.run_id)
            except Exception:
                current = None
            if current is None or (current.state in _RECOVERABLE and current.remote is None):
                threading.Thread(target=_deferred_resubmit_loop, args=(spec,), daemon=True).start()
            continue
        # Teardown/listing could not be confirmed (a possibly-live box). DON'T race it: defer with
        # observation bounded by the run wall deadline, then durably persist terminal success or failure.
        _log.warning(
            "deferring resubmit of %s: instance teardown unconfirmed; scheduling background retry",
            spec.run_id,
        )
        with contextlib.suppress(Exception):
            _append_run_log(
                spec.run_id,
                "control plane restart: instance teardown unconfirmed; "
                "deferring resubmit for reconciliation",
            )
        threading.Thread(target=_deferred_resubmit_loop, args=(spec,), daemon=True).start()
