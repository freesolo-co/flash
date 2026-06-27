"""FastAPI control plane for the managed Flash service."""

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
from flash.serve.export import export_adapter

from . import db
from ._locks import _DEPLOY_LOCKS, _deploy_lock
from ._runtime import (
    _RECOVERABLE,
    _charge_retry_loop,
    _charge_retry_startup,
    _reconcile_cost_loop,
    _worker_artifacts,
    recover_runs,
)

_DEPLOYABLE_STATES = {"done", "deployed"}
# cancelled/failed runs may still have per-step RL checkpoints streamed to HF
_CHECKPOINT_DEPLOYABLE_STATES = _DEPLOYABLE_STATES | {"cancelled", "failed"}
_SERVER_EXTRAS_HINT = "the control plane needs the server extras: pip install 'flash[server]'"

_log = logging.getLogger("flash.server")

__all__ = [
    "_DEPLOY_LOCKS",
    "_RECOVERABLE",
    "_charge_retry_loop",
    "_charge_retry_startup",
    "_deploy_lock",
    "_reconcile_cost_loop",
    "_worker_artifacts",
    "create_app",
    "deploy_adapter",
    "export_adapter",
    "get_status",
    "list_checkpoints",
    "recover_runs",
    "run_server",
    "serve_chat",
    "serve_chat_stream",
    "submit_job",
    "undeploy_adapter",
]


def _train_endpoint_names(*, include_terminal: bool) -> set[str]:
    """Return canonical RunPod training-endpoint names from the run registry."""
    from flash.providers.base import canonical_gpu
    from flash.providers.runpod.jobs import canonical_endpoint_name
    from flash.providers.runpod.train import _run_suffix, endpoint_name
    from flash.runner import TERMINAL_STATES

    names: set[str] = set()

    def _add(name: str | None) -> None:
        if name:
            names.add(canonical_endpoint_name(name))

    for row in db.all_runs():
        try:
            status = get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if not include_terminal and status.state in TERMINAL_STATES:
            continue
        _add((status.remote or {}).get("endpoint_name"))
        gpu = ((status.spec or {}).get("gpu") or {}).get("type")
        if gpu:
            with contextlib.suppress(Exception):
                _add(endpoint_name(canonical_gpu(gpu), _run_suffix(status.run_id)))
    return names


def _protected_train_endpoint_names() -> set[str]:
    return _train_endpoint_names(include_terminal=False)


def _known_train_endpoint_names() -> set[str]:
    return _train_endpoint_names(include_terminal=True)


def _reap_idle_endpoints_once(min_idle_s: float) -> int:
    """Sweep idle orphaned RunPod training endpoints. Returns count deleted."""
    from flash.providers.runpod.jobs import _sweep_idle_flash_endpoints

    return _sweep_idle_flash_endpoints(
        _protected_train_endpoint_names(),
        min_idle_s=min_idle_s,
        known=_known_train_endpoint_names(),
    )


async def _reap_idle_endpoints_loop() -> None:
    """Background loop: delete idle orphaned RunPod training endpoints."""
    interval = 600.0
    min_idle_s = 900.0
    _log.info(
        "idle-endpoint reaper started (sweep every %ds, reap after %ds idle)",
        int(interval),
        int(min_idle_s),
    )
    while True:
        await asyncio.sleep(interval)
        try:
            deleted = await asyncio.to_thread(_reap_idle_endpoints_once, min_idle_s)
            if deleted:
                _log.info("reaped %d idle RunPod endpoint(s) doing nothing", deleted)
        except asyncio.CancelledError:
            raise  # shutdown
        except Exception:
            _log.debug("idle-endpoint reaper sweep failed; retrying next cycle", exc_info=True)


# Aliased to _RECOVERABLE: a run is recoverable iff it may still own an in-flight worker.
_INSTANCE_OWNING_STATES = _RECOVERABLE


def _active_run_ids() -> set[str]:
    """Run ids that may still own a live training instance — protected from the orphan sweep.

    Passed as a callable so it is resolved AFTER the provider lists, closing the launch race."""
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
    """All run ids this plane has ever issued — scopes the sweep to our own boxes (multi-plane safety)."""
    return {row["run_id"] for row in db.all_runs()}


def _sweep_orphan_instances_once() -> int:
    """Sweep orphaned instance-provider workers left billing by finished/crashed runs. Returns count torn down."""
    from flash.providers import configured_providers

    torn = 0
    for prov in configured_providers():
        try:
            deleted = prov.sweep_orphans(
                active_labels=_active_run_ids, known_labels=_known_run_ids
            )
        except Exception:
            _log.warning(
                "instance orphan sweep failed for provider %r; retrying next cycle",
                getattr(prov, "name", prov),
                exc_info=True,
            )
            continue
        torn += len(deleted)
    return torn


async def _sweep_orphan_instances_loop() -> None:
    """Background loop: tear down orphaned Lambda instances left billing by finished/crashed runs."""
    interval = 600.0
    while True:
        await asyncio.sleep(interval)
        try:
            torn = await asyncio.to_thread(_sweep_orphan_instances_once)
            if torn:
                _log.info("swept %d orphaned instance-provider worker(s)", torn)
        except asyncio.CancelledError:
            raise  # shutdown
        except Exception:
            _log.debug("instance orphan sweep failed; retrying next cycle", exc_info=True)


def _instance_providers_configured() -> bool:
    """True when an instance-based provider (Lambda) is configured."""
    from flash.providers import available_providers

    return any(name in ("lambda",) for name in available_providers())


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
        from flash.server.billing_retry import charge_retry_enabled
        from flash.server.reconcile import reconcile_enabled

        check_run_preflight()
        recover_runs()
        # Background: retry charges that were pending/failed at shutdown (idempotent, non-blocking).
        startup_charge_task = (
            asyncio.create_task(_charge_retry_startup()) if charge_retry_enabled() else None
        )
        # Reconcile RunPod endpoint-slot quota so a crash can't leak slots permanently.
        with contextlib.suppress(Exception):
            from flash.providers.runpod.train.endpoints import reconcile_endpoint_slots

            reconcile_endpoint_slots()
        cost_task = asyncio.create_task(_reconcile_cost_loop()) if reconcile_enabled() else None
        charge_task = (
            asyncio.create_task(_charge_retry_loop()) if charge_retry_enabled() else None
        )
        reap_task = (
            asyncio.create_task(_reap_idle_endpoints_loop())
            if os.environ.get("RUNPOD_API_KEY")
            else None
        )
        sweep_task = (
            asyncio.create_task(_sweep_orphan_instances_loop())
            if _instance_providers_configured()
            else None
        )
        try:
            yield
        finally:
            for task in (startup_charge_task, cost_task, charge_task, reap_task, sweep_task):
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
