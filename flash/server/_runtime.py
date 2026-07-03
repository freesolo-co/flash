"""Background lifecycle helpers for the control plane: realized-cost reconciliation, run
recovery after a restart, and per-run log / worker-artifact access.

No fastapi dependency, so this module is safe to import at ``flash.server.app`` import time
(it must not pull in the optional server extras).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading

from flash.runner import adapter_prefix, get_status, runs_file_path
from flash.spec import JobSpec

from . import db

_log = logging.getLogger("flash.server")

# Run states that are still in flight and must be recovered after a control-plane restart.
_RECOVERABLE = {"queued", "provisioning", "running"}


async def _reconcile_cost_loop() -> None:
    """Background loop: periodically pull realized provider cost (COGS) for finished runs and
    report it to the freesolo backend for estimator accuracy. The provider billing calls are
    blocking urllib, so each sweep is offloaded to a thread; failures are swallowed and retried
    next cycle. Off entirely when FREESOLO_INTERNAL_KEY is unset (see reconcile_enabled)."""
    from flash.server.reconcile import reconcile_once

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


async def _charge_retry_startup() -> None:
    """Run ONE completion-charge recovery sweep off the startup critical path.

    The startup sweep must still happen promptly (the periodic loop sleeps a full interval before
    its first sweep, and recover_runs deliberately excludes terminal `done`, so a crash between the
    `done` write and the charge would otherwise leak revenue until the first periodic cycle). But it
    must NOT block the lifespan from `yield`-ing: with a backlog of pending/failed charges and a slow
    or down billing backend, each charge can wait the full billing timeout, turning a deploy/restart
    into minutes of unavailability. So the lifespan schedules this as a background task and the sweep
    runs in a thread (the charge is blocking urllib). Best-effort; cancelled cleanly at shutdown."""
    from flash.server.billing_retry import retry_completion_charges_once

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
    """Background loop: periodically re-charge completed runs whose customer charge was left
    pending/charging/failed by a transient backend blip (or a crash between the ``done`` write and
    the charge), so a finished-but-uncharged run never leaks revenue. The backend charge is
    idempotent by runId, so every retry is safe. Each sweep is offloaded to a thread (the charge is
    blocking urllib); failures are swallowed and retried next cycle. The interval IS the bounded
    backoff. Off entirely when FREESOLO_INTERNAL_KEY is unset (see charge_retry_enabled)."""
    from flash.server.billing_retry import retry_completion_charges_once

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
    """Append a timestamped note to a run's log so it surfaces in `flash log`."""
    import time

    with open(runs_file_path(run_id, ".log"), "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")


# A handle-less recovery resubmit is DEFERRED when this run's instance teardown can't be confirmed (a
# possibly-live Vast box that must not be raced by a second worker). The defer must not strand the run
# until the next control-plane restart — schedule a bounded background retry so it resubmits as soon as
# the phantom is gone / the listing recovers. After the budget it's left for the next restart.
_DEFERRED_RECOVERY_RETRY_S = 120.0
_DEFERRED_RECOVERY_MAX_RETRIES = 10


def _confirm_run_clear(spec) -> bool:
    """Force-reap this run's instance-provider label and report whether NO instance for it remains.

    A best-effort ``gc`` returns no error on an unconfirmed Vast teardown (``destroy_run_instances``
    yields an empty list, not a raise), so after the reap we ask each instance provider's optional
    ``run_instances_remaining`` for positive proof: ``[]`` == confirmed clear; a non-empty list (an
    instance is still present) or a RAISE (the provider can't ENUMERATE, so it can't prove clear) both
    yield ``False`` — the caller must not resubmit a second worker over a possibly-live box (Codex;
    mirrors the retry-loop MtzrH guard).

    A provider that does NOT implement ``run_instances_remaining`` (RunPod serverless self-reaps; Lambda
    today) is NOT a "can't confirm -> False" case: it has no standing-billing label to enumerate, so it
    is skipped and the run stays eligible to resubmit on the other providers' verdicts. This guard is
    therefore scoped to instance providers that DO expose the capability (Vast); only a provider that
    HAS the method and can't return a clean ``[]`` blocks the resubmit.

    BILLING-SAFETY: a provider that COULD have owned this run's lost create but is now
    UNCONFIGURABLE must also block the resubmit. A handle-less run's pre-handle non-idempotent create
    (Vast's ``PUT /asks``) may have left a phantom; if ``VAST_API_KEY`` was dropped before this restart,
    ``configured_providers()`` omits Vast, so iterating only the configured set returns "clear" and lets
    a second worker resubmit on another provider while the phantom keeps billing + writing the same HF
    prefix. We therefore ALSO fail closed for any provider recorded as available when the run was
    submitted (``submitted_instance_providers``) that owns the standing-instance capability yet can't be
    enumerated now. Scoping to the recorded set is what keeps this correct on planes that never
    configure Vast: such a run never recorded Vast, so its handle-less recovery is never blocked (it
    can't have left a Vast phantom)."""
    from flash.providers import INSTANCE_PROVIDERS, configured_providers, get_provider

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
    # An instance provider that WAS available at submit (so it could have taken the lost create) but is
    # NOT configurable now and owns the standing-instance capability can't be enumerated -> can't prove
    # clear -> fail closed. Already-configured providers were handled above; non-capability ones (Lambda)
    # have no standing label to leak. CRITICAL: do NOT wrap this in a broad ``suppress(Exception)`` — an
    # error loading the status, resolving a recorded provider, or reading the capability would otherwise
    # be swallowed and leave ``clear`` True, defeating the fail-closed intent (cursor). Every failure
    # inspecting the recorded set must instead make the guard CONSERVATIVE (block/defer the resubmit).
    for name in recorded or []:
        if name in configured or name not in INSTANCE_PROVIDERS:
            continue
        try:
            has_capability = getattr(get_provider(name), "run_instances_remaining", None) is not None
        except Exception:
            # Can't even resolve a RECORDED instance provider -> assume it could own a phantom -> fail
            # closed rather than declare clear.
            has_capability = True
        if has_capability:
            clear = False
    return clear


def _start_resubmit(spec) -> None:
    from flash.runner import _run_job_background

    with contextlib.suppress(Exception):
        _append_run_log(spec.run_id, "control plane restarted before provisioning; resubmitting")
    threading.Thread(target=_run_job_background, args=(spec,), daemon=True).start()


def _deferred_resubmit_loop(spec) -> None:
    """Background retry for a DEFERRED handle-less resubmit: re-confirm the reap on a bounded schedule
    and resubmit once it's clear, so a transient Vast listing failure / a phantom reaped by a later
    sweep doesn't leave the run stranded until the next control-plane restart."""
    import time

    from flash.runner import TERMINAL_STATES

    for _ in range(_DEFERRED_RECOVERY_MAX_RETRIES):
        time.sleep(_DEFERRED_RECOVERY_RETRY_S)
        with contextlib.suppress(Exception):
            st = get_status(spec.run_id)
            if st is not None and st.state in TERMINAL_STATES:
                return  # cancelled / failed / done meanwhile -> nothing to resubmit
        if _confirm_run_clear(spec):
            _start_resubmit(spec)
            return
    _log.warning(
        "giving up deferred resubmit of %s after %d retries; the next control-plane restart will retry",
        spec.run_id,
        _DEFERRED_RECOVERY_MAX_RETRIES,
    )


def _latest_worker_artifact_name(repo: str, prefix: str, phase: str, kind: str) -> str:
    """Newest worker artifact file under prefix.

    Workers historically wrote stable files (``console_rl.txt``), and newer workers may write
    attempt-scoped files (e.g. ``console_rl_attempt2.txt`` / ``error_rl_attempt2.txt``). On a retried run only the highest attempt is
    the real current evidence. Falls back to the legacy/default name when the repo can't be listed.
    """
    import re

    default = f"{kind}_{phase}.txt" if kind == "console" else f"{kind}_{phase}_attempt0.txt"
    try:
        from huggingface_hub import HfApi

        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            repo_id=repo, repo_type="dataset"
        )
    except Exception:
        return default
    pat = re.compile(
        rf"^{re.escape(prefix)}/{kind}_{re.escape(phase)}(?:_attempt(\d+))?\.txt$"
    )
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


def _worker_artifacts(spec) -> dict[str, str]:
    """The run's train-subprocess stdout + traceback, fetched from its HF artifact repo.

    The control-plane ``.log`` only carries orchestrator lines (and, on a terminal failure, a
    truncated tail of the worker console). The full ``console_<phase>.txt`` / ``error_<phase>.txt``
    the worker streams to HF are the real train stdout/traceback — but the repo is PRIVATE, so a
    user's own HF token 404s. We fetch them here with the OPERATOR ``HF_TOKEN`` (the control plane
    already holds it) so ``flash log`` shows the real worker output regardless of run
    state and without the user needing repo access. Best-effort: a missing file / no repo yields {}.
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
    for name in (
        _latest_worker_artifact_name(repo, prefix, spec.phase, "console"),
        _latest_error_artifact_name(repo, prefix, spec.phase),
    ):
        try:
            path = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=f"{prefix}/{name}",
                token=os.environ.get("HF_TOKEN"),
                # The worker appends to console/error files across the run, so a cached copy goes
                # stale; force a fresh pull (matches other HF artifact readers, e.g.
                # flash/providers/runpod/jobs.py:make_hf_text_reader).
                force_download=True,
            )
            # errors="replace": worker stdout can carry non-UTF-8 bytes (tracebacks, progress bars);
            # decode leniently so a single bad byte never drops the whole log on UnicodeDecodeError.
            with open(path, encoding="utf-8", errors="replace") as f:
                out[name] = f.read()
        except Exception:
            continue  # file not uploaded yet / not produced for this phase
    return out


def recover_runs() -> None:
    """Recover every in-flight run after a restart so a redeploy never loses a training session:
    re-attach to ``running`` jobs, and resubmit ``queued``/``provisioning`` runs that never reached
    a worker."""
    from flash.runner import (
        _gc_run_endpoints,
        _resolve_init_from_adapter,
        _status_org_id,
        _update,
        attach_run,
        get_status,
    )

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
    resubmit: list[JobSpec] = []
    for row in db.all_runs():
        known.add(row["run_id"])
        try:
            status = get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if status.state not in _RECOVERABLE:
            continue
        if status.remote:
            # Only handle-backed runs are kept by the sweep; a handle-less run is being
            # resubmitted, so its stale half-rented instance (if any) must NOT be shielded.
            active.add(status.run_id)
            threading.Thread(target=lambda rid=row["run_id"]: attach_run(rid), daemon=True).start()
        else:
            # No handle yet: the restart hit the submit->provisioning window, so no worker exists.
            # A spec that won't parse can never be resubmitted -> mark it terminally failed
            # (operator-visible, dropped from _RECOVERABLE so it isn't re-skipped every restart);
            # otherwise GC any half-made endpoint and resubmit from scratch.
            try:
                spec = JobSpec.from_dict(status.spec)
            except Exception as exc:
                _log.warning(
                    "marking run %s failed: persisted spec could not be parsed",
                    status.run_id,
                    exc_info=True,
                )
                detail = f"unrecoverable: persisted spec is malformed: {exc}"
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
                with contextlib.suppress(Exception):
                    gpu_type = (status.spec.get("gpu") or {}).get("type")
                    if gpu_type:
                        from flash.providers.runpod.train import terminate_endpoint

                        terminate_endpoint(gpu_type, status.run_id)
                continue
            with contextlib.suppress(Exception):
                _gc_run_endpoints(spec)
            try:
                owner_key_id = None
                with contextlib.suppress(Exception):
                    owner_key_id = db.run_owner(status.run_id)
                spec = _resolve_init_from_adapter(
                    spec,
                    owner_org_id=_status_org_id(status),
                    owner_key_id=owner_key_id,
                )
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
            resubmit.append(spec)
    # Reap orphaned per-run provider resources; each provider sweeps its own.
    from flash.providers import configured_providers

    for prov in configured_providers():
        with contextlib.suppress(Exception):
            prov.sweep_orphans(active_labels=active, known_labels=known)

    for spec in resubmit:
        _log.info("resubmitting run %s after control-plane restart", spec.run_id)
        # MtzrJ: a handle-less run hit the submit->provisioning window, so a NON-IDEMPOTENT instance
        # create (Vast's PUT /asks) may have been accepted while the response/handle was lost — a
        # phantom contract that bills and, worse, writes this run/seed's HF artifacts. The batch
        # sweep_orphans above reaps it only if it was VISIBLE then; _confirm_run_clear force-reaps this
        # run's label across the instance providers RIGHT BEFORE relaunching and verifies nothing for the
        # run remains, so a phantom that surfaced in the sweep->resubmit gap (Vast's instance list is
        # eventually consistent) can't get a SECOND worker writing the same seed-scoped artifacts.
        if _confirm_run_clear(spec):
            _start_resubmit(spec)
            continue
        # Teardown/listing could not be confirmed (a possibly-live box). DON'T race it: defer, and
        # schedule a bounded background retry so the run resubmits as soon as the phantom is gone / the
        # listing recovers — rather than stranding it until the next control-plane restart.
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
