"""Background lifecycle helpers: cost reconciliation, run recovery, and log access."""

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

_RECOVERABLE = {"queued", "provisioning", "running"}


async def _reconcile_cost_loop() -> None:
    """Periodically pull realized provider COGS for finished runs and report to the backend."""
    from flash.server.reconcile import reconcile_once

    interval = 3600.0
    while True:
        await asyncio.sleep(interval)
        try:
            reported = await asyncio.to_thread(reconcile_once)
            if reported:
                _log.info("reconciled realized cost for %d run(s)", reported)
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.debug("realized-cost reconcile sweep failed; retrying next cycle", exc_info=True)


async def _charge_retry_startup() -> None:
    """Run one completion-charge recovery sweep at startup without blocking the lifespan yield."""
    from flash.server.billing_retry import retry_completion_charges_once

    stop = threading.Event()
    try:
        recovered = await asyncio.to_thread(retry_completion_charges_once, stop.is_set)
        if recovered:
            _log.info("recovered %d pending completion charge(s) at startup", recovered)
    except asyncio.CancelledError:
        stop.set()  # task.cancel() only cancels the await; signal thread to stop between runs
        raise
    except Exception:
        _log.debug(
            "startup completion-charge sweep failed; periodic loop will retry", exc_info=True
        )


async def _charge_retry_loop() -> None:
    """Periodically re-charge completed runs whose customer charge was left pending/failed."""
    from flash.server.billing_retry import retry_completion_charges_once

    interval = 300.0
    while True:
        await asyncio.sleep(interval)
        stop = threading.Event()
        try:
            charged = await asyncio.to_thread(retry_completion_charges_once, stop.is_set)
            if charged:
                _log.info("recovered %d pending completion charge(s) on retry", charged)
        except asyncio.CancelledError:
            stop.set()
            raise
        except Exception:
            _log.debug("completion-charge retry sweep failed; retrying next cycle", exc_info=True)


def _append_run_log(run_id: str, message: str) -> None:
    """Append a timestamped note to a run's log so it surfaces in `flash status --logs`."""
    import time

    with open(runs_file_path(run_id, ".log"), "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")


def _worker_artifacts(spec) -> dict[str, str]:
    """Fetch worker console/error logs from the private HF artifact repo using the operator token."""
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
                force_download=True,  # worker appends across run; cached copy goes stale
            )
            # errors="replace": worker stdout can carry non-UTF-8 bytes from tracebacks/progress bars
            with open(path, encoding="utf-8", errors="replace") as f:
                out[name] = f.read()
        except Exception:
            continue
    return out


def recover_runs() -> None:
    """Re-attach running jobs, resume multi-seed runs, and resubmit unprovisioned runs after restart."""
    from flash.runner import (
        _gc_run_endpoints,
        _run_job_background,
        _update,
        attach_run,
        resume_run,
    )

    active: set[str] = set()
    resubmit: list[JobSpec] = []  # deferred until after orphan sweep to avoid racing fresh allocation
    for row in db.all_runs():
        try:
            status = get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if status.state not in _RECOVERABLE:
            continue
        if status.remote:
            active.add(status.run_id)
            threading.Thread(target=lambda rid=row["run_id"]: attach_run(rid), daemon=True).start()
        elif status.resume_seed_index is not None:
            active.add(status.run_id)
            threading.Thread(target=lambda rid=row["run_id"]: resume_run(rid), daemon=True).start()
        else:
            # No handle: restart hit the submit→provisioning window; GC any half-made endpoint and resubmit.
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
                # Crashed run may have registered a RunPod endpoint before dying; tear it down now
                # rather than waiting for the 15-min idle grace period.
                with contextlib.suppress(Exception):
                    gpu_type = (status.spec.get("gpu") or {}).get("type")
                    if gpu_type:
                        from flash.providers.runpod.train import terminate_endpoint

                        terminate_endpoint(gpu_type, status.run_id)
                continue
            with contextlib.suppress(Exception):
                _gc_run_endpoints(spec)
            resubmit.append(spec)
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
