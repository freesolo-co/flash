"""FastAPI control plane for the managed Flash service.

This is the operator-side component. It holds the
provider credentials (``RUNPOD_API_KEY``, ``HF_TOKEN``, ``PRIME_API_KEY`` —
the worker needs the last to ``prime env install`` the run's Prime Hub env) and
exposes the full run lifecycle to clients that authenticate with their freesolo API
key (verified against the freesolo backend) — clients never see provider credentials.

Run state truth stays in the runner's JSON files; SQLite (server/db.py) holds
keys and run ownership. Runs the server owns are recovered on startup by re-attaching
to their persisted RunPod job handles.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import weakref

from flash import __version__
from flash.catalog import public_model_rows
from flash.runner import (
    adapter_prefix,
    cancel_run,
    get_status,
    mark_deployed,
    mark_undeployed,
    new_run_id,
    runs_file_path,
    submit_job,
)
from flash.schema import ConfigError, spec_from_dict
from flash.serve.deploy import ServingError, deploy_adapter, undeploy_adapter
from flash.serve.deploy import chat as serve_chat
from flash.spec import JobSpec, coerce_bool

from . import auth, db

_RECOVERABLE = {"queued", "provisioning", "running"}
# Run states that have produced a downloadable adapter artifact.
_DEPLOYABLE_STATES = {"done", "deployed"}
_SERVER_EXTRAS_HINT = "the control plane needs the server extras: pip install 'flash[server]'"

_log = logging.getLogger("flash.server")


async def _reconcile_cost_loop() -> None:
    """Background loop: periodically pull realized provider cost (COGS) for finished runs and
    report it to the freesolo backend for estimator accuracy. The provider billing calls are
    blocking urllib, so each sweep is offloaded to a thread; failures are swallowed and retried
    next cycle. Off entirely when FREESOLO_INTERNAL_KEY is unset (see reconcile_enabled)."""
    from flash.server.reconcile import reconcile_once

    interval = float(os.environ.get("FLASH_RECONCILE_INTERVAL_S", "3600"))
    while True:
        await asyncio.sleep(interval)
        with contextlib.suppress(Exception):
            reported = await asyncio.to_thread(reconcile_once)
            if reported:
                _log.info("reconciled realized cost for %d run(s)", reported)


def _reverse_charge_best_effort(*, token: str, run_id: str, charge: dict) -> None:
    """Refund a run's pre-flight charge after its submit failed; never raises (logs on failure
    so the refund error can't mask the submit error). Idempotent by ``run_id``."""
    try:
        from flash.server.billing import reverse_run_charge

        reverse_run_charge(token=token, run_id=run_id, charge=charge)
    except Exception:
        # A refund failure must NOT mask the submit error: log and swallow.
        _log.exception(
            "failed to reverse the pre-flight charge for run %s after a submit failure; "
            "the org may be charged for a run that never started — reconcile manually",
            run_id,
        )


class _RunLock:
    """A weak-referenceable mutex usable as a context manager.

    ``threading.Lock()`` returns a ``_thread.lock`` that does NOT support weak references,
    so it can't live in a WeakValueDictionary directly — wrap it in a tiny object that does
    (and acquire/release via ``with``).
    """

    __slots__ = ("__weakref__", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __enter__(self) -> _RunLock:
        self._lock.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


# Per-run lock serializing deploy vs undeploy: registration with the freesolo serving app
# is slow and runs OUTSIDE the status lock, so without this the two could interleave —
# a racing undeploy could leave a stale deployment record (registered with freesolo but
# unrecorded here, or vice-versa), or a deploy's cleanup of a raced finalize could clobber
# another. Serving is delegated to freesolo (scales to zero per base model), so there is no
# billable flash-side endpoint at stake — only the deployment record's consistency.
# WeakValueDictionary so an entry is dropped once no request holds the lock — the map
# can't grow unboundedly with one entry per distinct run_id over the server's lifetime.
_DEPLOY_LOCKS: weakref.WeakValueDictionary[str, _RunLock] = weakref.WeakValueDictionary()
_DEPLOY_LOCKS_GUARD = threading.Lock()


def _deploy_lock(run_id: str) -> _RunLock:
    # The returned lock must be held by the caller (a `with` block) to keep it alive; once
    # released and unreferenced, the weak entry is garbage-collected.
    with _DEPLOY_LOCKS_GUARD:
        lk = _DEPLOY_LOCKS.get(run_id)
        if lk is None:
            lk = _RunLock()
            _DEPLOY_LOCKS[run_id] = lk
        return lk


def _append_run_log(run_id: str, message: str) -> None:
    """Append a timestamped note to a run's log so it surfaces in `flash logs`."""
    import time

    with open(runs_file_path(run_id, ".log"), "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")


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
                # `_gc_run_endpoints` guards against). The orphan sweep below won't reap it —
                # RunPod's `sweep_orphans` is a no-op — so do a best-effort RunPod GC HERE.
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
    # Reap orphaned per-run instances (Vast bills until destroyed); each provider sweeps its own.
    from flash.providers import configured_providers

    for prov in configured_providers():
        with contextlib.suppress(Exception):
            prov.sweep_orphans(active_labels=active)

    for spec in resubmit:
        _log.info("resubmitting run %s after control-plane restart", spec.run_id)
        with contextlib.suppress(Exception):
            _append_run_log(spec.run_id, "control plane restarted before provisioning; resubmitting")
        threading.Thread(target=_run_job_background, args=(spec,), daemon=True).start()


def create_app():
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError as exc:
        raise RuntimeError(_SERVER_EXTRAS_HINT) from exc
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        from flash.providers.preflight import check_run_preflight
        from flash.server.reconcile import reconcile_enabled

        check_run_preflight()  # operator credentials: fail fast, before serving anyone
        recover_runs()
        # Periodic realized-cost reconciliation (estimator accuracy), only when the operator
        # internal key is configured. First (and only) asyncio task in the server.
        cost_task = asyncio.create_task(_reconcile_cost_loop()) if reconcile_enabled() else None
        try:
            yield
        finally:
            if cost_task is not None:
                cost_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cost_task

    app = FastAPI(title="Flash Control Plane", version=__version__, lifespan=lifespan)

    def require_key(authorization: str | None = Header(default=None)) -> dict:
        key = auth.authenticate(authorization)
        if key is None:
            raise HTTPException(
                status_code=401,
                detail="invalid or missing API key; log in with `flash login` using your "
                "freesolo API key",
            )
        return key

    def owned_run(run_id: str, key: dict):
        """Load a run's status iff `key` owns it; 404 otherwise (don't leak existence)."""
        if db.run_owner(run_id) != key["id"]:
            raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
        try:
            return get_status(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/health")
    def health():
        return {"ok": True, "service": "flash", "version": __version__}

    @app.get("/v1/me")
    def me(key: dict = Depends(require_key)):
        return {
            "key_prefix": key["key_prefix"],
            "email": key["email"],
            "created_at": key["created_at"],
        }

    @app.get("/v1/models")
    def models(_: dict = Depends(require_key)):
        return {"models": public_model_rows()}

    @app.post("/v1/envs")
    def publish_env(payload: dict, key: dict = Depends(require_key)):
        # Publish a client-built verifiers env package to the MANAGED Prime account (this control
        # plane's PRIME_API_KEY), namespaced per identity and PRIVATE — so users never need their
        # own Prime account. The client just packages local source and uploads it here.
        from flash.server import envs

        # Default to "" only when the key is missing/None — pass a present-but-falsy
        # non-string (0, False, []) THROUGH so publish_package's type checks reject it with
        # the right 400, instead of `or ""` silently coercing it to a valid-looking empty string.
        _pkg = payload.get("package_b64")
        _name = payload.get("name")
        try:
            slug = envs.publish_package(
                package_b64="" if _pkg is None else _pkg,
                name="" if _name is None else _name,
                # Robust bool parse: JSON `"is_new": "false"`/`"0"` must NOT become True
                # (plain bool() is truthy for any non-empty string). Defaults True when absent.
                is_new=coerce_bool(payload.get("is_new", True)),
                key=key,
            )
        except envs.EnvPublishError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        return {"id": slug}

    def _parse_spec(payload: dict, run_id: str) -> JobSpec:
        spec_raw = payload.get("spec") or {}
        env_raw = spec_raw.get("environment") or {}
        if env_raw.get("path"):
            raise HTTPException(
                status_code=400,
                detail="local environment paths are not supported on the managed service; "
                "publish the environment to the Prime Hub (`flash env push`), then reference it "
                'by its Hub id (`[environment] id = "owner/name"`)',
            )
        try:
            return spec_from_dict(spec_raw, run_id=run_id)
        except (ConfigError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/runs")
    def create_run(
        payload: dict,
        key: dict = Depends(require_key),
        authorization: str | None = Header(default=None),
    ):
        spec = _parse_spec(payload, run_id=new_run_id())
        dry_run = bool(payload.get("dry_run", False))
        # Charge the pre-flight estimate to the user's org BEFORE accepting the run, so it's gated
        # on balance and never starts unpaid. Skipped for --dry-run and the internal service
        # identity (no user org to bill).
        billed = not dry_run and key.get("key_prefix") != "internal"
        token = (authorization or "").removeprefix("Bearer ").strip()
        charge: dict = {}
        if billed:
            from flash.server.billing import BillingError as _BillingError
            from flash.server.billing import charge_run_estimate

            try:
                charge = charge_run_estimate(token=token, spec=spec)
            except _BillingError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
            except ValueError as exc:
                # Unpriceable config (e.g. uncapped SFT whose env can't be counted): 400 with a
                # fixable message, not a 500.
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        # record_run is inside the try so any post-charge failure (SQLite, submit) reverses the
        # debit — the org must never pay for a run that didn't start.
        try:
            db.record_run(spec.run_id, key["id"])
            status = submit_job(spec, dry_run=dry_run, background=True)
        except Exception as exc:
            db.delete_run(spec.run_id)  # idempotent: a no-op if record_run never landed
            # The run never started — reverse the charge so the org isn't billed for it.
            if billed:
                _reverse_charge_best_effort(token=token, run_id=spec.run_id, charge=charge)
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return status.to_dict()

    @app.get("/v1/runs")
    def list_runs(key: dict = Depends(require_key)):
        out = []
        for row in db.runs_for_key(key["id"]):
            try:
                out.append(get_status(row["run_id"]).to_dict())
            except FileNotFoundError:
                continue
        return {"runs": out}

    @app.get("/v1/runs/{run_id}")
    def run_status(run_id: str, key: dict = Depends(require_key)):
        status = owned_run(run_id, key)
        return status.to_dict()

    @app.get("/v1/runs/{run_id}/logs")
    def run_logs(run_id: str, offset: int = 0, key: dict = Depends(require_key)):
        status = owned_run(run_id, key)
        log_path = runs_file_path(run_id, ".log")
        chunk, end = "", max(0, offset)
        if os.path.exists(log_path):
            with open(log_path) as f:
                f.seek(end)
                chunk = f.read()
                end = f.tell()
        return {"run_id": run_id, "logs": chunk, "offset": end, "state": status.state}

    @app.post("/v1/runs/{run_id}/cancel")
    def cancel(run_id: str, key: dict = Depends(require_key)):
        owned_run(run_id, key)
        return cancel_run(run_id).to_dict()

    @app.post("/v1/runs/{run_id}/deploy")
    def deploy(run_id: str, payload: dict | None = None, key: dict = Depends(require_key)):
        payload = payload or {}
        # Serialize deploy vs undeploy (and a second deploy) for this run: registration
        # with the freesolo serving app runs outside the status lock, so without this they
        # could interleave and leave the serving record and the control plane inconsistent.
        with _deploy_lock(run_id):
            status = owned_run(run_id, key)
            spec = JobSpec.from_dict(status.spec)
            dry_run = bool(payload.get("dry_run", False))
            if not dry_run and status.state not in _DEPLOYABLE_STATES:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"run {run_id} is {status.state!r}; only finished runs with "
                        "trained adapter artifacts can be deployed"
                    ),
                )
            # Legacy runs persisted before [train].hf_repo was mandatory rehydrate with an
            # empty hf_repo; without this guard freesolo serving cannot locate the adapter
            # artifacts (the per-run HF dataset repo). Reject early with a clear 409.
            if not dry_run and not spec.train.hf_repo:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"run {run_id} has no [train].hf_repo (legacy run); its adapter artifacts "
                        "cannot be located, so it cannot be deployed"
                    ),
                )
            mode = payload.get("mode", "dev")
            # The state the run must still be in for this deploy to finalize — a CAS guard so
            # a /cancel (NOT serialized by the deploy lock) that terminalized the run can't be
            # silently overwritten by the deployment record.
            prev_state = status.state
            try:
                dep = deploy_adapter(
                    run_id=run_id,
                    model=spec.model,
                    hf_repo=spec.train.hf_repo,
                    adapter_prefix=adapter_prefix(spec),
                    gpu_name=spec.gpu.type,
                    mode=mode,
                    idle_timeout_s=int(payload.get("idle_timeout_s", 300)),
                    dry_run=dry_run,
                    # a run trained with thinking serves with thinking (per-run parity)
                    thinking=spec.thinking,
                )
            except ServingError as exc:
                # The serving backend rejected the registration or was unreachable. This is an
                # upstream/gateway failure, not a flash bug, so surface a clean 502 with the
                # real reason instead of letting httpx escape as an unhandled 500 + traceback.
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                raise
            if not dry_run:
                # Record the deployment. The CAS no-ops only if a /cancel raced finalization
                # — then the adapter we just registered is orphaned, so deregister it and
                # report the conflict instead of a bogus 200.
                marked = mark_deployed(run_id, dep.to_dict(), expect_state=prev_state)
                if marked.state != "deployed":
                    with contextlib.suppress(Exception):
                        undeploy_adapter(run_id)
                    raise HTTPException(
                        status_code=409,
                        detail=f"run {run_id} became {marked.state!r} during deploy; aborted",
                    )
            return dep.to_dict()

    @app.delete("/v1/runs/{run_id}/deploy")
    def undeploy(run_id: str, key: dict = Depends(require_key)):
        # Same per-run lock as deploy: an undeploy must not interleave with an in-flight
        # deploy's provisioning/finalization.
        with _deploy_lock(run_id):
            status = owned_run(run_id, key)
            try:
                deleted = undeploy_adapter(run_id)
            except ServingError as exc:
                # A serving-backend failure (unreachable / non-404 error) is an upstream/gateway
                # problem, not a flash bug — surface a clean 502 with the real reason (mirrors the
                # deploy handler) instead of letting the ServingError escape as an unhandled 500.
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            # dev mode is scale-to-zero: the serve endpoint is created only on the first
            # chat, so an empty deletion just means it was never warmed — still a clean
            # undeploy. always-on provisions the endpoint at deploy time, so an empty
            # deletion there is a transient RunPod failure that must NOT hide a
            # still-billable endpoint (surface 502 so the user retries).
            dev_mode = (status.deployment or {}).get("mode", "dev") == "dev"
            if status.deployment and (deleted or dev_mode):
                mark_undeployed(run_id)
            elif status.deployment and not deleted:
                raise HTTPException(
                    status_code=502,
                    detail=f"could not delete the serving endpoint for {run_id}; it may still "
                    "be running — retry `flash undeploy`",
                )
            return {"run_id": run_id, "deleted_endpoints": deleted}

    @app.get("/v1/deployments")
    def deployments(key: dict = Depends(require_key)):
        out = []
        for row in db.runs_for_key(key["id"]):
            try:
                status = get_status(row["run_id"])
            except FileNotFoundError:
                continue
            if status.deployment and status.deployment.get("state") not in (
                "undeployed",
                "dry_run",
            ):
                out.append(status.to_dict())
        return {"deployments": out}

    @app.post("/v1/runs/{run_id}/chat")
    def chat(run_id: str, payload: dict, key: dict = Depends(require_key)):
        status = owned_run(run_id, key)
        spec = JobSpec.from_dict(status.spec)
        deployment = status.deployment or {}
        # A cancelled run's serve endpoint was torn down at cancel time; never let a
        # chat recreate it (closes the window before cancel marks the deployment
        # inactive, and covers a teardown that deleted nothing).
        if status.state == "cancelled":
            raise HTTPException(
                status_code=409, detail=f"run {run_id} was cancelled; redeploy is not allowed"
            )
        # Chat must ride an explicit deployment (with its cost controls), not
        # implicitly provision a serving endpoint that /v1/deployments cannot see.
        if deployment.get("state") in (None, "undeployed", "dry_run"):
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} has no active deployment; `flash deploy {run_id}` first",
            )
        # Legacy run with no artifact repo (mirrors the /deploy guard): a run that never had a
        # [train].hf_repo was never registered with freesolo serving, so reject early with a
        # clear 409 instead of an opaque downstream inference error.
        if not spec.train.hf_repo:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} has no [train].hf_repo (legacy run); its adapter cannot be served",
            )
        try:
            return serve_chat(
                run_id=run_id,
                messages=payload.get("messages") or [],
                temperature=float(payload.get("temperature") or 0.0),
                max_tokens=int(payload.get("max_tokens") or 512),
                # a run trained with thinking serves with thinking (per-run parity)
                thinking=spec.thinking,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"inference failure: {exc}") from exc

    return app


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(_SERVER_EXTRAS_HINT) from exc
    uvicorn.run(create_app(), host=host, port=port)
