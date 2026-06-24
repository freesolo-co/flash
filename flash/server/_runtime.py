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
        with contextlib.suppress(Exception):
            reported = await asyncio.to_thread(reconcile_once)
            if reported:
                _log.info("reconciled realized cost for %d run(s)", reported)


def _append_run_log(run_id: str, message: str) -> None:
    """Append a timestamped note to a run's log so it surfaces in `flash status --logs`."""
    import time

    with open(runs_file_path(run_id, ".log"), "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")


def _worker_artifacts(spec) -> dict[str, str]:
    """The run's train-subprocess stdout + traceback, fetched from its HF artifact repo.

    The control-plane ``.log`` only carries orchestrator lines (and, on a terminal failure, a
    truncated tail of the worker console). The full ``console_<phase>.txt`` / ``error_<phase>.txt``
    the worker streams to HF are the real train stdout/traceback — but the repo is PRIVATE, so a
    user's own HF token 404s. We fetch them here with the OPERATOR ``HF_TOKEN`` (the control plane
    already holds it) so ``flash status --logs`` shows the real worker output regardless of run
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
    for name in (f"console_{spec.phase}.txt", f"error_{spec.phase}.txt"):
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
    re-attach to ``running`` jobs, resume multi-seed runs across the inter-seed gap, and resubmit
    ``queued``/``provisioning`` runs that never reached a worker."""
    from flash.runner import (
        _gc_run_endpoints,
        _run_job_background,
        _update,
        attach_run,
        resume_run,
    )

    active: set[str] = set()
    # Deferred until after the orphan sweep so a half-rented instance from a crashed pre-handle
    # attempt is reaped without racing the resubmit's fresh allocation.
    resubmit: list[JobSpec] = []
    for row in db.all_runs():
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
        elif status.resume_seed_index is not None:
            # Restarted between seeds: resume the remaining seeds, preserving the finished ones.
            active.add(status.run_id)
            threading.Thread(target=lambda rid=row["run_id"]: resume_run(rid), daemon=True).start()
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
            resubmit.append(spec)
    # Reap orphaned per-run provider resources; each provider sweeps its own.
    from flash.providers import configured_providers

    for prov in configured_providers():
        with contextlib.suppress(Exception):
            prov.sweep_orphans(active_labels=active)

    for spec in resubmit:
        _log.info("resubmitting run %s after control-plane restart", spec.run_id)
        with contextlib.suppress(Exception):
            _append_run_log(
                spec.run_id, "control plane restarted before provisioning; resubmitting"
            )
        threading.Thread(target=_run_job_background, args=(spec,), daemon=True).start()
