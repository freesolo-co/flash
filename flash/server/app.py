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
    """Training-endpoint names derived from the run registry, in the CANONICAL (bare ``flash-...``)
    form — the reaper canonicalizes the ``live-flash-...`` names RunPod lists before comparing, so
    one form per name suffices. Includes both the run's persisted handle name and the name re-derived
    from its spec, so a run is covered even in the submit -> handle-persisted window.

    ``include_terminal=False`` yields the PROTECTED set (live, non-terminal runs only) — endpoints
    the reaper must never delete, even when momentarily idle between seeds. ``include_terminal=True``
    yields the KNOWN set (every run this plane has a record of) — used to SCOPE the reaper to this
    plane's own endpoints (multi-plane safety; see ``_sweep_idle_flash_endpoints``)."""
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
    """Endpoint names tied to a LIVE (non-terminal) run — never reaped (see ``_train_endpoint_names``)."""
    return _train_endpoint_names(include_terminal=False)


def _known_train_endpoint_names() -> set[str]:
    """Endpoint names for EVERY run this plane has a record of — the reaper's multi-plane scope."""
    return _train_endpoint_names(include_terminal=True)


def _reap_idle_endpoints_once(min_idle_s: float) -> int:
    """One run-aware sweep of idle, orphaned RunPod training endpoints. Returns count deleted."""
    from flash.providers.runpod.jobs import _sweep_idle_flash_endpoints

    return _sweep_idle_flash_endpoints(
        _protected_train_endpoint_names(),
        min_idle_s=min_idle_s,
        known=_known_train_endpoint_names(),
    )


async def _reap_idle_endpoints_loop() -> None:
    """Background loop: proactively delete idle, orphaned RunPod training endpoints (workers doing
    nothing that still hold worker quota) so they don't linger between quota errors. Run-aware and
    graced (see ``_sweep_idle_flash_endpoints``); the blocking RunPod calls are offloaded to a
    thread, and a failed sweep is logged and retried next cycle."""
    interval = 600.0  # sweep every 10 min
    min_idle_s = 900.0  # only reap an endpoint idle for >= 15 min (well past any cold start)
    # Startup heartbeat: the reaper otherwise logs nothing unless it deletes something, so an
    # operator can't tell a silently-stalled reaper from a healthy one with nothing to reap. One
    # INFO line at boot confirms the task is alive and pins its cadence.
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
            raise  # shutdown: let the lifespan's task.cancel() propagate, don't swallow it
        except Exception:
            _log.debug("idle-endpoint reaper sweep failed; retrying next cycle", exc_info=True)


# Run states that may still OWN a live, billing training instance, so their provider instances must
# be PROTECTED from the orphan sweep. Deliberately EXCLUDES ``deployed``: a run only reaches
# ``deployed`` after it went ``done`` (the seed loop's ``finally`` already tore every training
# instance down), so a deployed run owns no training worker — keeping it in the protection set would
# instead SHIELD a genuine leaked instance under its prefix from the sweep (the very thing the sweep
# exists to reap). Terminal states are excluded for the same reason. This is exactly ``_RECOVERABLE``
# — a run is recoverable on restart iff it may still have an in-flight worker — so it is ALIASED
# (one source of truth) to keep the two protection sets from silently drifting apart.
_INSTANCE_OWNING_STATES = _RECOVERABLE


def _active_run_ids() -> set[str]:
    """Run ids of every run that may still own a live training instance — the set whose provider
    instances must be PROTECTED from the periodic orphan sweep below. The instance providers'
    ``sweep_orphans`` re-derives each instance-label prefix from a run id via ``run_label_prefix``,
    so it wants raw run ids (unlike ``_protected_train_endpoint_names``, which yields RunPod endpoint
    *names*).

    Why this is a safe protection set with no idle grace: a run's status is flipped to an
    instance-owning state BEFORE its first instance is ever launched (``_run_training`` writes
    ``running`` ahead of ``_submit_seed_supervised``), and the launched instance is torn down BEFORE
    the run can leave these states for ``done``/``deployed``/terminal (the provider lifecycle's
    ``finally``). So a billed instance exists ONLY while its run is in this set — ownership is a
    deterministic name->run mapping, not the noisy idle signal the RunPod reaper must grace. The
    sweep passes this function itself (a callable) so the set is read AFTER the provider lists, which
    closes the launch race — see ``_sweep_orphan_instances_once``. (Startup recovery in
    ``recover_runs`` deliberately uses a NARROWER set — only handle-backed/resume runs — because it
    is simultaneously RESUBMITTING handle-less runs and must reap their stale half-rented instances;
    in-lifetime we instead protect every instance-owning run.)"""
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
    """Every run id THIS control plane has a record of, in ANY state — the universe of instances it
    may attribute to itself. Passed to the instance providers' ``sweep_orphans`` as ``known_labels``
    so the sweep reaps only boxes whose label maps to one of our own runs.

    This is the multi-plane safety guard. Two control planes sharing one provider account carry
    DISJOINT run ids (server-assigned ``flash-<ts>-<rand>``). Without it, each plane's sweep sees the
    OTHER's live boxes, finds their run ids absent from its ``_active_run_ids``, and reaps them — the
    two planes mutually execute each other's running training instances every sweep. Scoping the reap
    to ``known_labels`` makes a plane skip any box whose run id it has never issued. Read straight off
    the run registry (no ``get_status`` per row needed — only the id matters), and like
    ``_active_run_ids`` it is passed as a CALLABLE so it is resolved AFTER the provider lists.

    Trade-off (deliberate): a box whose run record this plane has fully LOST (e.g. a wiped state dir)
    is no longer auto-reaped — it is indistinguishable from another plane's box, so erring toward not
    destroying it is the safe choice; reclaim such strays out of band. Single-plane production is
    unaffected: the durable registry holds every run it launched, so every genuine orphan stays
    reapable."""
    return {row["run_id"] for row in db.all_runs()}


def _sweep_orphan_instances_once() -> int:
    """One run-aware sweep of orphaned instance-provider workers — Lambda instances whose run
    finished or crashed without the per-run ``finally`` tearing them down. Returns the count torn
    down. Dispatched to every configured provider; RunPod's ``sweep_orphans`` is a no-op (its
    serverless endpoints carry no standing per-run billing and are handled by the idle reaper).

    ``_active_run_ids`` is passed as a CALLABLE, not a precomputed set, so each instance provider
    resolves the live-run protection set AFTER it has listed its instances. That ordering closes the
    launch race: any instance already present in the list had its run's status row committed before
    the instance was launched, so it is guaranteed to be in the set read post-listing — a run that
    started a worker concurrently with this sweep can never be mis-reaped as a phantom orphan. (The
    instance APIs expose no creation timestamp, so this post-listing read — not an age grace — is
    what makes it airtight.)"""
    from flash.providers import configured_providers

    torn = 0
    for prov in configured_providers():
        try:
            deleted = prov.sweep_orphans(
                active_labels=_active_run_ids, known_labels=_known_run_ids
            )
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
    """Background loop: proactively tear down orphaned instance-provider workers (billed instances
    left by finished/crashed runs that the per-run ``finally`` teardown missed) so they stop
    billing without waiting for the next control-plane restart. Covers all instance-billed
    providers (Lambda, Vast). This is the in-lifetime counterpart of the instance providers'
    startup ``sweep_orphans`` (``recover_runs``) — the instance analogue of
    ``_reap_idle_endpoints_loop`` for RunPod. Blocking provider calls are offloaded to a thread; a
    failed sweep is logged and retried next cycle."""
    interval = 600.0  # sweep every 10 min (matches the RunPod idle reaper)
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
    """True when an instance-based provider (Lambda or Vast) is configured on this plane, so the
    periodic instance orphan sweep is worth running. RunPod-only planes skip it — RunPod has no
    standing per-run billing to reap between restarts (its idle reaper covers warm endpoints)."""
    from flash.providers import INSTANCE_PROVIDERS, available_providers

    return any(name in INSTANCE_PROVIDERS for name in available_providers())


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

        check_run_preflight()  # operator credentials: fail fast, before serving anyone
        recover_runs()
        # Recover completion-time customer charges left pending/failed by a transient blip or a
        # crash between the `done` write and the charge. recover_runs deliberately excludes terminal
        # `done`, so those would otherwise leak revenue; this startup sweep catches them promptly.
        # Idempotent by runId on the backend, so it can't double-charge. Scheduled as a BACKGROUND
        # task (not awaited before `yield`): with a backlog of pending charges and a slow/down billing
        # backend, each charge can wait the full billing timeout, so awaiting it inline would delay
        # accepting traffic by minutes. Best-effort; cancelled cleanly at shutdown below.
        startup_charge_task = (
            asyncio.create_task(_charge_retry_startup()) if charge_retry_enabled() else None
        )
        # Reconcile the shared RunPod endpoint-slot quota against the live endpoint list so a
        # crash can't leak slots permanently (no-op without an internal key). Best-effort.
        with contextlib.suppress(Exception):
            from flash.providers.runpod.train.endpoints import reconcile_endpoint_slots

            reconcile_endpoint_slots()
        # Periodic realized-cost reconciliation (estimator accuracy), only when the operator
        # internal key is configured.
        cost_task = asyncio.create_task(_reconcile_cost_loop()) if reconcile_enabled() else None
        # Periodic completion-charge retry: re-charge any run left pending/failed by a transient blip
        # so it can't leak revenue. Same internal-key gate as the charge itself.
        charge_task = (
            asyncio.create_task(_charge_retry_loop()) if charge_retry_enabled() else None
        )
        # Periodic idle-endpoint reaper: proactively delete RunPod training endpoints doing
        # nothing (orphans from finished/crashed runs) so workers don't linger holding quota.
        # Only when this plane manages RunPod (its API key is configured).
        reap_task = (
            asyncio.create_task(_reap_idle_endpoints_loop())
            if os.environ.get("RUNPOD_API_KEY")
            else None
        )
        # Periodic instance orphan sweep: proactively tear down Lambda instances left billing by
        # finished/crashed runs (the in-lifetime counterpart of their startup sweep_orphans). Only
        # when an instance provider is configured — RunPod-only planes have nothing standing to reap.
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
