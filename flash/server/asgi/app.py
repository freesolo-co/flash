"""FastAPI control plane for the managed Flash service.

It owns provider credentials, run recovery, and route wiring. Imports stay light: FastAPI and route
modules load only inside ``create_app`` and ``run_server`` for the optional server extra.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import threading
import time

import flash.runner.lifecycle.status as run_status
import flash.server.platform.deployment_jobs as deployment_jobs
import flash.server.platform.runtime as server_runtime
from flash import __version__
from flash._internal.background import Supervisor
from flash.server.platform import db, plane_lease

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


class _LeaderWork:
    """Recovery and background loops. Only the control-plane lease holder runs them."""

    def __init__(self, app) -> None:
        self.app = app
        self.deployment_cleanup_stop = threading.Event()
        self.startup_report_stop = threading.Event()
        # (task, tolerate_errors): the status replay may fail at shutdown, the loops may not.
        self.tasks: list[tuple[asyncio.Task, bool]] = []
        # A takeover's recovery thread; shutdown waits for it, since a thread cannot be cancelled.
        self.recovery: asyncio.Future | None = None

    def recover(self) -> None:
        from flash.server.routes import serving_completion

        # Reserve the durable-backlog scanner before recovery can fill its bounded lane.
        with self.app.state.training_jobs.reserve(
            ("scan", "runs"), "recovery", delay=30.0
        ) as dispatch:
            server_runtime.recover_runs()
            dispatch(server_runtime.recover_pending_runs)
        serving_completion.recover_deployments()

    def start(self) -> None:
        """Start the background loops; call on the event loop after recover()."""
        from flash.server.billing.retry import charge_retry_enabled
        from flash.server.domain.ops.repo_cleanup import repo_cleanup_enabled
        from flash.server.routes import serving_completion

        deployment_jobs.start_deployment_job(
            serving_completion.run_deployment_cleanup_worker,
            self.deployment_cleanup_stop,
            background=True,
        )
        # replay one persisted status at a time in the background. synchronous per-item delivery
        # prevents a historical backlog from filling the shared reporter ahead of live updates.
        self._spawn(
            asyncio.to_thread(serving_completion.replay_status_reports, self.startup_report_stop),
            tolerate_errors=True,
        )
        if charge_retry_enabled():
            # retry completion charges missed by transient failure or a crash after done, then keep
            # re-charging any run left pending/failed by a transient blip so it can't leak revenue.
            # run in background so billing timeouts cannot delay startup; backend runId makes it
            # idempotent. Same internal-key gate as the charge itself.
            self._spawn(server_runtime._charge_retry_startup())
            self._spawn(server_runtime._charge_retry_loop())
        if _instance_providers_configured():
            self._spawn(_sweep_orphan_instances_loop())
        # periodic artifact gc: delete aged (>7d), undeployed run prefixes inside the per-environment
        # hf repos (<artifact namespace>/flashrun-*) so old runs' checkpoints/adapters don't pile up
        # against the org's storage quota. only on a plane with an operator `hf_token` (it deletes
        # operator-owned repos); fails closed on any live-set uncertainty. see
        # flash.server.domain.ops.repo_cleanup.
        if repo_cleanup_enabled():
            self._spawn(server_runtime._repo_cleanup_loop())

    def _spawn(self, coroutine, *, tolerate_errors: bool = False) -> None:
        self.tasks.append((asyncio.create_task(coroutine), tolerate_errors))

    def signal_stop(self) -> None:
        self.deployment_cleanup_stop.set()
        self.startup_report_stop.set()

    async def cancel(self) -> None:
        for task, tolerate_errors in self.tasks:
            task.cancel()
            ignored = (
                (asyncio.CancelledError, Exception)
                if tolerate_errors
                else (asyncio.CancelledError,)
            )
            with contextlib.suppress(*ignored):
                await task


def _stop_plane() -> None:
    """Shut this server down gracefully; the container's restart policy brings it back."""
    signal.raise_signal(signal.SIGTERM)


async def _lead_when_free(work: _LeaderWork) -> None:
    """Wait for the previous plane to exit, then take over recovery and the loops."""
    await plane_lease.acquire()
    _log.info("control-plane lease acquired; starting recovery and background work")
    # Submissions stay queued until recovery and its orphan sweep finish: a launch racing the sweep
    # has no persisted handle yet, so the sweep would reap its instance. The recovery scanner that
    # recover() dispatches launches the runs queued meanwhile.
    work.recovery = asyncio.ensure_future(asyncio.to_thread(work.recover))
    try:
        await asyncio.shield(work.recovery)
    except asyncio.CancelledError:
        raise
    except Exception:
        # A lone plane fails startup when recovery fails; a takeover must not keep serving as if
        # it led. Stop, and let the restart recover from a clean start.
        _log.critical("control-plane recovery failed after lease takeover; stopping", exc_info=True)
        _stop_plane()
        return
    work.app.state.plane_standby = False
    work.start()


def create_app():
    try:
        from fastapi import FastAPI

        from flash.server.routes import (
            adapter_download,
            custom,
            env_files,
            envs,
            meta,
            runs,
            serving,
        )
    except ImportError as exc:
        raise RuntimeError(_SERVER_EXTRAS_HINT) from exc
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def managed_lifespan(app):
        from flash.providers.core.preflight import check_run_preflight
        from flash.runner.lifecycle.reporting import _open_status_reporter

        check_run_preflight()  # operator credentials: fail fast, before serving anyone
        deployment_jobs._open_deployment_jobs()
        _open_status_reporter()
        work = _LeaderWork(app)
        waiter = None
        if plane_lease.try_acquire():
            # The usual case: no other plane on this state directory. Recover before serving.
            work.recover()
            work.start()
        else:
            # A deploy started this plane while the previous one drains. Serve HTTP now; new runs
            # stay queued for the holder's recovery scan until its exit hands the lease over.
            _log.warning(
                "another control plane holds %s; serving without recovery until it exits",
                plane_lease.lease_path(),
            )
            app.state.plane_standby = True
            waiter = asyncio.create_task(_lead_when_free(work))
        try:
            yield
        finally:
            if waiter is not None:
                waiter.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await waiter
            if work.recovery is not None:
                # The lease is released after shutdown; never hand it over mid-recovery.
                with contextlib.suppress(Exception):
                    await work.recovery
            work.signal_stop()
            app.state.training_jobs.close()
            await work.cancel()
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
            if await asyncio.to_thread(owner.close, 5.0):
                plane_lease.release()
            else:
                # A task still in a provider call may be launching or supervising a run. Keep the
                # lease so no other plane recovers that run; the kernel drops it when we exit.
                _log.warning(
                    "training tasks still in blocking calls at shutdown deadline; "
                    "keeping the control-plane lease until this process exits"
                )

    app = FastAPI(title="Flash Control Plane", version=__version__, lifespan=lifespan)
    app.include_router(meta.router)
    app.include_router(envs.router)
    app.include_router(env_files.router)
    app.include_router(runs.router)
    app.include_router(custom.router)
    app.include_router(serving.router)
    app.include_router(adapter_download.router)
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
