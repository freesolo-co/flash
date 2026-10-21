"""FastAPI control plane for the managed Flash service.

It owns provider credentials, run recovery, and route wiring. Imports stay light: FastAPI and route
modules load only inside ``create_app`` and ``run_server`` for the optional server extra.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time

import flash.runner.lifecycle.status as run_status
import flash.server.platform.deployment_jobs as deployment_jobs
import flash.server.platform.runtime as server_runtime
from flash import __version__
from flash._internal.background import Supervisor
from flash.server.platform import db

_SERVER_EXTRAS_HINT = "the control plane needs the server extras: pip install 'flash[server]'"

_log = logging.getLogger("flash.server")


# states that may still own billed training instances and must be protected from orphan sweeps.
# alias server_runtime._RECOVERABLE; deployed and terminal runs own no worker and must not shield leaks.
_INSTANCE_OWNING_STATES = server_runtime._RECOVERABLE


def _active_run_ids() -> set[str]:
    """Return run ids that may still own a live training instance.

    Status enters an owning state before launch and leaves only after teardown. Pass this callable
    so
    providers read the set after listing instances, closing the concurrent-launch race.
    """
    ids: set[str] = set()
    for row in db.all_runs():
        try:
            status = run_status.get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if status.state in _INSTANCE_OWNING_STATES:
            ids.add(status.run_id)
    return ids


def _known_run_ids() -> set[str]:
    """Return every run id recorded by this control plane, in any state.

    ``known_labels`` prevents planes sharing a provider account from reaping each other's instances.
    Lost registry rows are deliberately spared because they are indistinguishable from another
    plane.
    """
    return {row["run_id"] for row in db.all_runs()}


def _sweep_orphan_instances_once() -> int:
    """Run one sweep of orphaned instance-provider workers and return the teardown count.

    Pass active ids as a callable so providers list first and read protection state afterward;
    instance APIs expose no creation timestamp for a reliable age grace.
    """
    from flash.providers.core.registry import configured_providers

    torn = 0
    for prov in configured_providers():
        try:
            deleted = prov.sweep_orphans(active_labels=_active_run_ids, known_labels=_known_run_ids)
        except Exception:
            # One provider's API blip / outage must not skip the others — and must NOT be silent
            # (the loop docstring promises failures are logged + retried next cycle), so a
            # persistent failure (bad creds, signature mismatch) is visible instead of looking
            # like a healthy sweep reaping nothing.
            _log.warning(
                "instance orphan sweep failed for provider %r; retrying next cycle",
                getattr(prov, "name", prov),
                exc_info=True,
            )
            continue
        torn += len(deleted)
    return torn


async def _sweep_orphan_instances_loop() -> None:
    """Periodically remove orphaned Modal workers left by finished or crashed runs."""
    interval = 600.0
    while True:
        await asyncio.sleep(interval)
        try:
            torn = await asyncio.to_thread(_sweep_orphan_instances_once)
            if torn:
                _log.info("swept %d orphaned instance-provider worker(s)", torn)
        except asyncio.CancelledError:
            raise  # shutdown: let the lifespan's task.cancel() propagate, don't swallow it
        except Exception:
            _log.debug("instance orphan sweep failed; retrying next cycle", exc_info=True)


def _instance_providers_configured() -> bool:
    """True when an instance-based provider is configured on this plane, so the"""
    from flash.providers.core.registry import INSTANCE_PROVIDERS, available_providers

    return any(name in INSTANCE_PROVIDERS for name in available_providers())


def create_app():
    try:
        from fastapi import FastAPI

        from flash.server.routes import envs, meta, runs, serving, serving_completion, teacher
    except ImportError as exc:
        raise RuntimeError(_SERVER_EXTRAS_HINT) from exc
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def managed_lifespan(app):
        from flash.providers.core.preflight import check_run_preflight
        from flash.runner.lifecycle.reporting import _open_status_reporter
        from flash.server.billing.retry import charge_retry_enabled

        check_run_preflight()  # operator credentials: fail fast, before serving anyone
        db.recover_teacher_request_ledger()
        deployment_jobs._open_deployment_jobs()
        _open_status_reporter()
        # Reserve the durable-backlog scanner before recovery can fill its bounded lane.
        with app.state.training_jobs.reserve(("scan", "runs"), "recovery", delay=30.0) as dispatch:
            server_runtime.recover_runs()
            dispatch(server_runtime.recover_pending_runs)
        serving_completion.recover_deployments()
        deployment_cleanup_stop = threading.Event()
        deployment_jobs.start_deployment_job(
            serving_completion.retry_deployment_cleanup,
            deployment_cleanup_stop,
            background=True,
        )
        # replay one persisted status at a time in the background. synchronous per-item delivery
        # prevents a historical backlog from filling the shared reporter ahead of live updates.
        startup_report_stop = threading.Event()
        startup_report_task = asyncio.create_task(
            asyncio.to_thread(serving_completion.replay_status_reports, startup_report_stop)
        )
        # retry completion charges missed by transient failure or a crash after done.
        # run in background so billing timeouts cannot delay startup; backend runId makes it
        # idempotent.
        startup_charge_task = (
            asyncio.create_task(server_runtime._charge_retry_startup())
            if charge_retry_enabled()
            else None
        )
        # Periodic completion-charge retry: re-charge any run left pending/failed by a transient blip
        # so it can't leak revenue. Same internal-key gate as the charge itself.
        charge_task = (
            asyncio.create_task(server_runtime._charge_retry_loop())
            if charge_retry_enabled()
            else None
        )
        sweep_task = (
            asyncio.create_task(_sweep_orphan_instances_loop())
            if _instance_providers_configured()
            else None
        )
        # periodic artifact gc: delete aged (>7d), undeployed run prefixes inside the per-environment
        # hf repos (<artifact namespace>/flashrun-*) so old runs' checkpoints/adapters don't pile up
        # against the org's storage quota. only on a plane with an operator `hf_token` (it deletes
        # operator-owned repos); fails closed on any live-set uncertainty. see
        # flash.server.domain.ops.repo_cleanup.
        from flash.server.domain.ops.repo_cleanup import repo_cleanup_enabled

        cleanup_task = (
            asyncio.create_task(server_runtime._repo_cleanup_loop())
            if repo_cleanup_enabled()
            else None
        )
        try:
            yield
        finally:
            deployment_cleanup_stop.set()
            app.state.training_jobs.close()
            startup_report_stop.set()
            startup_report_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await startup_report_task
            for task in (
                startup_charge_task,
                charge_task,
                sweep_task,
                cleanup_task,
            ):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            shutdown_deadline = time.monotonic() + 15.0
            with contextlib.suppress(Exception):
                if not await asyncio.to_thread(deployment_jobs._wait_for_deployment_jobs, 10.0):
                    _log.warning("deployment jobs still running at shutdown deadline")
            with contextlib.suppress(Exception):
                from flash.runner.lifecycle.reporting import _shutdown_status_reporter

                remaining = max(0.0, shutdown_deadline - time.monotonic())
                await asyncio.to_thread(_shutdown_status_reporter, remaining, close=True)

    @asynccontextmanager
    async def lifespan(app):
        owner = Supervisor()
        app.state.training_jobs = owner
        try:
            with owner.bind():
                async with managed_lifespan(app):
                    yield
        finally:
            if not await asyncio.to_thread(owner.close, 5.0):
                _log.warning("training tasks still in blocking calls at shutdown deadline")

    app = FastAPI(title="Flash Control Plane", version=__version__, lifespan=lifespan)
    app.include_router(meta.router)
    app.include_router(envs.router)
    app.include_router(runs.router)
    app.include_router(serving.router)
    app.include_router(teacher.router)
    return app


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(_SERVER_EXTRAS_HINT) from exc
    from flash.providers.core.preflight import require_operator_config

    # Before uvicorn, not just in the lifespan. The lifespan copy stays -- it is what covers a plane
    # built through create_app() by another ASGI server -- but a PreflightError raised there is an
    # unhandled ASGI startup exception, so the operator's actual problem arrives under ~20 lines of
    # starlette/contextlib frames. Running it here lets __main__ print the message on its own.
    #
    # The refusing half only: this is the call that repeats, and the lifespan is where the advisory
    # summary belongs, so a booted plane logs it once.
    require_operator_config()
    uvicorn.run(create_app(), host=host, port=port)
