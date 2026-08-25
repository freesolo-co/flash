"""FastAPI control plane for the managed Flash service.

It owns provider credentials, run recovery, and route wiring. Imports stay light: FastAPI and route
modules load only inside ``create_app`` and ``run_server`` for the optional server extra.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time

from flash import __version__
from flash.runner.lifecycle.status import get_status
from flash.runner.lifecycle.submit import prepare_job, submit_job
from flash.runner.results.checkpoints import list_checkpoints
from flash.serve.deployment.deploy import (
    adapter_alias_target,
    deploy_adapter,
    deployment_record,
    undeploy_adapter,
)
from flash.serve.deployment.deploy import chat as serve_chat
from flash.serve.deployment.deploy import chat_sse as serve_chat_sse
from flash.serve.deployment.deploy import chat_stream as serve_chat_stream
from flash.serve.deployment.export import export_adapter
from flash.server.platform import db
from flash.server.platform.locks import _DEPLOY_LOCKS, _deploy_lock
from flash.server.platform.runtime import (
    _RECOVERABLE,
    _charge_retry_loop,
    _charge_retry_startup,
    _reconcile_cost_loop,
    _repo_cleanup_loop,
    _worker_artifacts,
    recover_runs,
)

# Run states that have produced a downloadable adapter artifact.
_DEPLOYABLE_STATES = {"done", "deployed"}
_SERVER_EXTRAS_HINT = "the control plane needs the server extras: pip install 'flash[server]'"

_log = logging.getLogger("flash.server")
_DEPLOYMENT_JOBS_LOCK = threading.Lock()
_DEPLOYMENT_JOBS: set[threading.Thread] = set()
_DEPLOYMENT_JOBS_ACCEPTING = True


class DeploymentJobStartError(RuntimeError):
    pass


# Symbols re-exported through this module. The route handlers look these up on
# ``flash.server.asgi.app`` at call time (``_app.<name>``) so a test that patches ``app.<name>``
# is honored; listing them here also keeps them from reading as unused imports.
__all__ = [
    "_DEPLOY_LOCKS",
    "_RECOVERABLE",
    "DeploymentJobStartError",
    "_charge_retry_loop",
    "_charge_retry_startup",
    "_deploy_lock",
    "_reconcile_cost_loop",
    "_repo_cleanup_loop",
    "_worker_artifacts",
    "adapter_alias_target",
    "create_app",
    "deploy_adapter",
    "deployment_record",
    "export_adapter",
    "get_status",
    "list_checkpoints",
    "prepare_job",
    "recover_runs",
    "run_server",
    "serve_chat",
    "serve_chat_sse",
    "serve_chat_stream",
    "start_deployment_job",
    "submit_job",
    "undeploy_adapter",
]


def _open_deployment_jobs() -> None:
    global _DEPLOYMENT_JOBS_ACCEPTING
    with _DEPLOYMENT_JOBS_LOCK:
        _DEPLOYMENT_JOBS_ACCEPTING = True


def _run_deployment_job(target, args, kwargs) -> None:
    try:
        target(*args, **kwargs)
    finally:
        with _DEPLOYMENT_JOBS_LOCK:
            _DEPLOYMENT_JOBS.discard(threading.current_thread())


def _wait_for_deployment_jobs(timeout: float) -> bool:
    global _DEPLOYMENT_JOBS_ACCEPTING
    deadline = time.monotonic() + timeout
    with _DEPLOYMENT_JOBS_LOCK:
        _DEPLOYMENT_JOBS_ACCEPTING = False
    while True:
        with _DEPLOYMENT_JOBS_LOCK:
            jobs = tuple(_DEPLOYMENT_JOBS)
        if not jobs:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        jobs[0].join(remaining)


def start_deployment_job(target, *args, **kwargs) -> bool:
    """Start a deployment lifecycle job.

    Returns True when the job ran synchronously (test mode), False when it was started in a
    background thread.
    """
    if os.environ.get("FLASH_DEPLOY_SYNC") == "1":
        target(*args, **kwargs)
        return True
    thread = threading.Thread(
        target=_run_deployment_job,
        args=(target, args, kwargs),
        daemon=True,
    )
    with _DEPLOYMENT_JOBS_LOCK:
        if not _DEPLOYMENT_JOBS_ACCEPTING:
            raise DeploymentJobStartError("deployment jobs are shutting down")
        _DEPLOYMENT_JOBS.add(thread)
        try:
            thread.start()
        except Exception as exc:
            _DEPLOYMENT_JOBS.discard(thread)
            raise DeploymentJobStartError(str(exc)) from exc
    return False


# states that may still own billed training instances and must be protected from orphan sweeps.
# alias _RECOVERABLE; deployed and terminal runs own no worker and must not shield leaks.
_INSTANCE_OWNING_STATES = _RECOVERABLE


def _active_run_ids() -> set[str]:
    """Return run ids that may still own a live training instance.

    Status enters an owning state before launch and leaves only after teardown. Pass this callable
    so
    providers read the set after listing instances, closing the concurrent-launch race.
    """
    ids: set[str] = set()
    for row in db.all_runs():
        try:
            status = get_status(row["run_id"])
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
    """Periodically tear down orphaned workers for every instance provider.

    This is the in-lifetime counterpart of the startup sweep in ``recover_runs``. Blocking provider
    calls run in a thread, and a failed sweep is logged and retried next cycle.
    """
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
    """Return whether this plane has any configured instance provider to sweep."""
    from flash.providers.core.registry import INSTANCE_PROVIDERS, available_providers

    return any(name in INSTANCE_PROVIDERS for name in available_providers())


def create_app():
    try:
        from fastapi import FastAPI

        from flash.server.routes import envs, meta, runs, serving, teacher
    except ImportError as exc:
        raise RuntimeError(_SERVER_EXTRAS_HINT) from exc
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        from flash.providers.core.preflight import check_run_preflight
        from flash.runner.lifecycle.reporting import _open_status_reporter
        from flash.server.billing.retry import charge_retry_enabled
        from flash.server.domain.ops.reconcile import reconcile_enabled

        check_run_preflight()  # operator credentials: fail fast, before serving anyone
        db.recover_teacher_request_ledger()
        _open_deployment_jobs()
        _open_status_reporter()
        recover_runs()
        serving.recover_deployments()
        # replay one persisted status at a time in the background. synchronous per-item delivery
        # prevents a historical backlog from filling the shared reporter ahead of live updates.
        startup_report_stop = threading.Event()
        startup_report_task = asyncio.create_task(
            asyncio.to_thread(serving.replay_status_reports, startup_report_stop)
        )
        # retry completion charges missed by transient failure or a crash after done.
        # run in background so billing timeouts cannot delay startup; backend runId makes it
        # idempotent.
        startup_charge_task = (
            asyncio.create_task(_charge_retry_startup()) if charge_retry_enabled() else None
        )
        # periodic realized-cost reconciliation runs only when the operator internal key is configured.
        cost_task = asyncio.create_task(_reconcile_cost_loop()) if reconcile_enabled() else None
        # periodic completion-charge retry recharges any run left pending or failed by a transient
        # blip so it cannot leak revenue. it uses the same internal-key gate as the charge itself.
        charge_task = asyncio.create_task(_charge_retry_loop()) if charge_retry_enabled() else None
        # periodically tear down orphaned runpod pods, lambda instances, and vast instances.
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

        cleanup_task = asyncio.create_task(_repo_cleanup_loop()) if repo_cleanup_enabled() else None
        try:
            yield
        finally:
            startup_report_stop.set()
            startup_report_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await startup_report_task
            for task in (
                startup_charge_task,
                cost_task,
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
                if not await asyncio.to_thread(_wait_for_deployment_jobs, 10.0):
                    _log.warning("deployment jobs still running at shutdown deadline")
            with contextlib.suppress(Exception):
                from flash.runner.lifecycle.reporting import _shutdown_status_reporter

                remaining = max(0.0, shutdown_deadline - time.monotonic())
                await asyncio.to_thread(_shutdown_status_reporter, remaining, close=True)

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
