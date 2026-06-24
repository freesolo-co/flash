"""FastAPI control plane for the managed Flash service.

This is the operator-side component. It holds the provider credentials
(``RUNPOD_API_KEY``, ``HF_TOKEN``, and environment source tokens) and exposes the
full run lifecycle to clients that authenticate with their freesolo API key
(verified against the freesolo backend) — clients never see provider credentials.

Run state truth stays in the runner's JSON files; SQLite (server/db.py) holds
keys and run ownership. Runs the server owns are recovered on startup by re-attaching
to their persisted RunPod job handles.

The HTTP routes live in cohesive ``flash.server.routes`` modules; this module wires them
into the FastAPI app, holds the background lifecycle helpers, and re-exports the service
symbols that the route handlers (and the test-suite monkeypatches) resolve through it.

Importing this module must stay light: fastapi (the optional ``[server]`` extra) is imported
only inside ``create_app`` / ``run_server``, which raise ``_SERVER_EXTRAS_HINT`` when it is
absent. The route and ``_deps`` modules import fastapi, so they too are imported lazily there.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os

from flash import __version__
from flash.runner import get_status, submit_job
from flash.runner.checkpoints import list_checkpoints
from flash.serve.deploy import chat as serve_chat
from flash.serve.deploy import chat_stream as serve_chat_stream
from flash.serve.deploy import deploy_adapter, undeploy_adapter

from . import db
from ._locks import _DEPLOY_LOCKS, _deploy_lock
from ._runtime import _RECOVERABLE, _reconcile_cost_loop, _worker_artifacts, recover_runs

# Run states that have produced a downloadable adapter artifact.
_DEPLOYABLE_STATES = {"done", "deployed"}
# A specific intermediate checkpoint can also be deployed from a run that stopped mid-RL
# (cancelled/failed): the per-step adapter was already streamed to HF, so it serves even though
# the run never sealed a final adapter. `dry_run` is excluded — it never trained.
_CHECKPOINT_DEPLOYABLE_STATES = _DEPLOYABLE_STATES | {"cancelled", "failed"}
_SERVER_EXTRAS_HINT = "the control plane needs the server extras: pip install 'flash[server]'"

_log = logging.getLogger("flash.server")

# Symbols re-exported through this module. The route handlers look these up on
# ``flash.server.app`` at call time (``_app.<name>``) so a test that patches ``app.<name>``
# is honored; listing them here also keeps them from reading as unused imports.
__all__ = [
    "_DEPLOY_LOCKS",
    "_RECOVERABLE",
    "_deploy_lock",
    "_reconcile_cost_loop",
    "_worker_artifacts",
    "create_app",
    "deploy_adapter",
    "get_status",
    "list_checkpoints",
    "recover_runs",
    "run_server",
    "serve_chat",
    "serve_chat_stream",
    "submit_job",
    "undeploy_adapter",
]


def _protected_train_endpoint_names() -> set[str]:
    """Training-endpoint names that must NEVER be reaped: every endpoint tied to a LIVE
    (non-terminal) run, in both the bare ``flash-...`` and SDK ``live-flash-...`` forms.

    Derived from the run registry so the reaper can't delete a run that's merely idle between
    jobs/seeds. Includes both the run's persisted handle name and the name re-derived from its
    spec, so a run is protected even in the submit -> handle-persisted provisioning window.
    """
    from flash.providers.base import canonical_gpu
    from flash.providers.runpod.train import _run_suffix, endpoint_name
    from flash.runner import TERMINAL_STATES

    names: set[str] = set()

    def _protect(name: str | None) -> None:
        if name:
            names.add(name)
            names.add(f"live-{name}")

    for row in db.all_runs():
        try:
            status = get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if status.state in TERMINAL_STATES:
            continue
        _protect((status.remote or {}).get("endpoint_name"))
        gpu = ((status.spec or {}).get("gpu") or {}).get("type")
        if gpu:
            with contextlib.suppress(Exception):
                _protect(endpoint_name(canonical_gpu(gpu), _run_suffix(status.run_id)))
    return names


def _reap_idle_endpoints_once(min_idle_s: float) -> int:
    """One run-aware sweep of idle, orphaned RunPod training endpoints. Returns count deleted."""
    from flash.providers.runpod.jobs import _sweep_idle_flash_endpoints

    return _sweep_idle_flash_endpoints(_protected_train_endpoint_names(), min_idle_s=min_idle_s)


async def _reap_idle_endpoints_loop() -> None:
    """Background loop: proactively delete idle, orphaned RunPod training endpoints (workers doing
    nothing that still hold worker quota) so they don't linger between quota errors. Run-aware and
    graced (see ``_sweep_idle_flash_endpoints``); the blocking RunPod calls are offloaded to a
    thread, and a failed sweep is logged and retried next cycle."""
    interval = 600.0  # sweep every 10 min
    min_idle_s = 900.0  # only reap an endpoint idle for >= 15 min (well past any cold start)
    while True:
        await asyncio.sleep(interval)
        try:
            deleted = await asyncio.to_thread(_reap_idle_endpoints_once, min_idle_s)
            if deleted:
                _log.info("reaped %d idle RunPod endpoint(s) doing nothing", deleted)
        except asyncio.CancelledError:
            raise  # shutdown: let the lifespan's task.cancel() propagate, don't swallow it
        except Exception:
            _log.debug("idle-endpoint reaper sweep failed; retrying next cycle", exc_info=True)


def create_app():
    try:
        from fastapi import FastAPI

        from flash.server.routes import envs, meta, runs, serving
    except ImportError as exc:
        raise RuntimeError(_SERVER_EXTRAS_HINT) from exc
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        from flash.providers.preflight import check_run_preflight
        from flash.server.reconcile import reconcile_enabled

        check_run_preflight()  # operator credentials: fail fast, before serving anyone
        recover_runs()
        # Reconcile the shared RunPod endpoint-slot quota against the live endpoint list so a
        # crash can't leak slots permanently (no-op without an internal key). Best-effort.
        with contextlib.suppress(Exception):
            from flash.providers.runpod.train.endpoints import reconcile_endpoint_slots

            reconcile_endpoint_slots()
        # Periodic realized-cost reconciliation (estimator accuracy), only when the operator
        # internal key is configured.
        cost_task = asyncio.create_task(_reconcile_cost_loop()) if reconcile_enabled() else None
        # Periodic idle-endpoint reaper: proactively delete RunPod training endpoints doing
        # nothing (orphans from finished/crashed runs) so workers don't linger holding quota.
        # Only when this plane manages RunPod (its API key is configured).
        reap_task = (
            asyncio.create_task(_reap_idle_endpoints_loop())
            if os.environ.get("RUNPOD_API_KEY")
            else None
        )
        try:
            yield
        finally:
            for task in (cost_task, reap_task):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    app = FastAPI(title="Flash Control Plane", version=__version__, lifespan=lifespan)
    app.include_router(meta.router)
    app.include_router(envs.router)
    app.include_router(runs.router)
    app.include_router(serving.router)
    return app


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(_SERVER_EXTRAS_HINT) from exc
    uvicorn.run(create_app(), host=host, port=port)
