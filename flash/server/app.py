"""FastAPI control plane for the managed Flash service.

This is the operator-side component. It holds the provider credentials
(``RUNPOD_API_KEY``, ``HF_TOKEN``, and environment source tokens) and exposes the
full run lifecycle to clients that authenticate with their freesolo API key
(verified against the freesolo backend) — clients never see provider credentials.

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
from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS
from flash.runner import (
    adapter_prefix,
    attach_checkpoint_deployment,
    cancel_run,
    get_status,
    mark_deployed,
    mark_undeployed,
    new_run_id,
    runs_file_path,
    submit_job,
)
from flash.runner.checkpoints import checkpoint_adapter_prefix, list_checkpoints
from flash.schema import ConfigError, spec_from_dict
from flash.serve.deploy import ServingError, deploy_adapter, undeploy_adapter
from flash.serve.deploy import chat as serve_chat
from flash.serve.deploy import chat_stream as serve_chat_stream
from flash.spec import JobSpec

from . import auth, db

_RUNTIME_SECRET_KEYS = DEFAULT_RUNTIME_SECRET_KEYS
_RECOVERABLE = {"queued", "provisioning", "running"}
# Run states that have produced a downloadable adapter artifact.
_DEPLOYABLE_STATES = {"done", "deployed"}
# A specific intermediate checkpoint can also be deployed from a run that stopped mid-RL
# (cancelled/failed): the per-step adapter was already streamed to HF, so it serves even though
# the run never sealed a final adapter. `dry_run` is excluded — it never trained.
_CHECKPOINT_DEPLOYABLE_STATES = _DEPLOYABLE_STATES | {"cancelled", "failed"}
_SERVER_EXTRAS_HINT = "the control plane needs the server extras: pip install 'flash[server]'"

_log = logging.getLogger("flash.server")


def _resolve_deploy_step(run_id: str, spec, raw_step) -> int | None:
    """Validate an optional deploy ``step`` against the run's published checkpoints.

    Returns the integer step to deploy, or ``None`` when no step was requested (deploy the
    final adapter). Raises ``HTTPException(400)`` for a malformed step and ``HTTPException(404)``
    — listing the available steps — when the run has no deployable checkpoint at that step."""
    if raw_step is None:
        return None
    from fastapi import HTTPException

    # Accept only an actual integer step — NOT a bool (True would coerce to step 1) and not a
    # non-integer float/string (40.9 / "40.9" must not silently round to a different checkpoint).
    want: int | None = None
    if isinstance(raw_step, bool):
        want = None
    elif isinstance(raw_step, int):
        want = raw_step
    elif isinstance(raw_step, float):
        want = int(raw_step) if raw_step.is_integer() else None
    elif isinstance(raw_step, str) and raw_step.strip().lstrip("-").isdigit():
        want = int(raw_step.strip())
    if want is None:
        raise HTTPException(status_code=400, detail=f"invalid checkpoint step: {raw_step!r}")
    checkpoints = list_checkpoints(spec)
    if any(c["step"] == want for c in checkpoints):
        return want
    available = ", ".join(str(c["step"]) for c in checkpoints) or "none"
    raise HTTPException(
        status_code=404,
        detail=f"run {run_id} has no deployable checkpoint at step {want} (available: {available})",
    )


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
                # stale; force a fresh pull (matches the other HF artifact readers, e.g.
                # providers/runpod/jobs.py make_hf_console_reader).
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


def create_app():
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse
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
        payload = {
            "kind": "internal" if key.get("auth_kind") == "internal" else "freesolo_api_key",
            "key_prefix": key["key_prefix"],
        }
        for field in (
            "email",
            "user_id",
            "org_id",
            "api_key_id",
            "training_agent_job_id",
            "project_id",
        ):
            if key.get(field):
                payload[field] = key[field]
        return payload

    @app.get("/v1/models")
    def models(_: dict = Depends(require_key)):
        return {"models": public_model_rows()}

    @app.post("/v1/envs")
    def publish_env(payload: dict, key: dict = Depends(require_key)):
        # Publish a client-built Freesolo environment package to the managed
        # environment repository. Users never need direct repository credentials.
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
                key=key,
            )
        except envs.EnvPublishError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        from flash.server.environment_registry import record_published_environment

        record_published_environment(slug=slug, name=str(_name), key=key)
        return {"id": slug}

    def _parse_spec(payload: dict, run_id: str) -> JobSpec:
        spec_raw = payload.get("spec") or {}
        env_raw = spec_raw.get("environment") or {}
        if env_raw.get("path"):
            raise HTTPException(
                status_code=400,
                detail="local environment paths are not supported on the managed service; "
                "publish the environment with `flash env push --name <name>`, then reference it "
                "by the returned environment id",
            )
        try:
            return spec_from_dict(spec_raw, run_id=run_id)
        except (ConfigError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _runtime_secrets(
        payload: dict, spec: JobSpec, *, require_environment_secrets: bool
    ) -> dict[str, str]:
        raw = payload.get("runtime_secrets") or {}
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="runtime_secrets must be a JSON object")
        allowed = set(_RUNTIME_SECRET_KEYS) | set(spec.environment.secrets)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=(
                    "unsupported runtime secret(s): "
                    f"{', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})"
                ),
            )
        out: dict[str, str] = {}
        for key, value in raw.items():
            if value is None:
                continue
            if not isinstance(value, str):
                raise HTTPException(
                    status_code=400, detail=f"runtime_secrets.{key} must be a string"
                )
            value = value.strip()
            if value:
                out[key] = value
        if require_environment_secrets:
            missing = sorted(set(spec.environment.secrets) - set(out))
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "missing runtime secret(s) required by [environment] secrets: "
                        f"{', '.join(missing)}"
                    ),
                )
        return out

    @app.post("/v1/runs")
    def create_run(
        payload: dict,
        key: dict = Depends(require_key),
    ):
        spec = _parse_spec(payload, run_id=new_run_id())
        dry_run = bool(payload.get("dry_run", False))
        runtime_secrets = _runtime_secrets(
            payload, spec, require_environment_secrets=not dry_run
        )
        # External user-key runs are charged only after training succeeds. Persist the org id
        # (non-secret) so the background runner can bill with the operator internal key at
        # completion; never persist the submitting user's API key.
        bill_on_completion = not dry_run and key.get("auth_kind") != "internal"
        billing_context = None
        if bill_on_completion:
            org_id = str(key.get("org_id") or "").strip()
            if not org_id:
                raise HTTPException(
                    status_code=400,
                    detail="org id is required to bill a completed training run",
                )
            billing_context = {"org_id": org_id}
        try:
            db.record_run(spec.run_id, key["id"])
            submit_kwargs = {"dry_run": dry_run, "background": True}
            if runtime_secrets:
                submit_kwargs["runtime_secrets"] = runtime_secrets
            if billing_context:
                submit_kwargs["billing_context"] = billing_context
            platform_context = {
                field: value
                for field, value in {
                    "org_id": key.get("org_id"),
                    "user_id": key.get("user_id"),
                    "api_key_id": key.get("api_key_id"),
                }.items()
                if value
            }
            if platform_context:
                submit_kwargs["platform_context"] = platform_context
            status = submit_job(spec, **submit_kwargs)
            from flash.server.run_registry import record_training_run

            record_training_run(status=status, key=key)
            from flash.envs.adapter import is_managed_environment_slug
            from flash.server.environment_registry import record_environment_use

            if is_managed_environment_slug(spec.environment.id):
                record_environment_use(slug=spec.environment.id, run_id=spec.run_id, key=key)
        except Exception as exc:
            db.delete_run(spec.run_id)  # idempotent: a no-op if record_run never landed
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
        return {
            "run_id": run_id,
            "logs": chunk,
            "offset": end,
            "state": status.state,
            "last_heartbeat": status.last_heartbeat,
            "gpu_status": status.gpu_status,
        }

    @app.get("/v1/runs/{run_id}/worker")
    def run_worker_output(run_id: str, key: dict = Depends(require_key)):
        # The full train-subprocess stdout/traceback, pulled from the run's HF artifact repo with
        # the operator token — the real worker output the offset-paged .log can't carry. Kept off
        # the hot /logs poll path (it hits HF) so streaming `--follow` stays fast; `--logs` calls
        # this once. Best-effort: {} when nothing's been uploaded yet.
        status = owned_run(run_id, key)
        return {"run_id": run_id, "worker": _worker_artifacts(JobSpec.from_dict(status.spec))}

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
            # Optional `step`: deploy a specific intermediate checkpoint instead of the run's
            # final adapter. We resolve it against what's actually on HF (the source of truth),
            # so a missing step 404s with the available list rather than 500ing at serve time.
            checkpoint_step = _resolve_deploy_step(run_id, spec, payload.get("step"))
            is_checkpoint = checkpoint_step is not None
            allowed_states = (
                _CHECKPOINT_DEPLOYABLE_STATES if is_checkpoint else _DEPLOYABLE_STATES
            )
            if not dry_run and status.state not in allowed_states:
                detail = (
                    f"run {run_id} is {status.state!r}; deploy a checkpoint only once the run "
                    "has finished or been cancelled"
                    if is_checkpoint
                    else f"run {run_id} is {status.state!r}; only finished runs with "
                    "trained adapter artifacts can be deployed"
                )
                raise HTTPException(status_code=409, detail=detail)
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
            # A checkpoint deploy serves the per-step adapter; otherwise the run's final adapter.
            deploy_prefix = (
                checkpoint_adapter_prefix(spec, checkpoint_step)
                if is_checkpoint
                else adapter_prefix(spec)
            )
            # The state the run must still be in for this deploy to finalize — a CAS guard so
            # a /cancel (NOT serialized by the deploy lock) that terminalized the run can't be
            # silently overwritten by the deployment record.
            prev_state = status.state
            try:
                dep = deploy_adapter(
                    run_id=run_id,
                    model=spec.model,
                    hf_repo=spec.train.hf_repo,
                    adapter_prefix=deploy_prefix,
                    gpu_name=spec.gpu.type,
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
            dep_dict = dep.to_dict()
            if is_checkpoint:
                dep_dict["checkpoint_step"] = checkpoint_step
            if not dry_run:
                if is_checkpoint and status.state not in _DEPLOYABLE_STATES:
                    # Deploying a checkpoint of a run that stopped mid-RL (cancelled/failed):
                    # attach the serving deployment but KEEP the run's terminal training state
                    # — flipping it to `deployed` would erase the outcome and make undeploy
                    # wrongly restore it to `done`.
                    attach_checkpoint_deployment(run_id, dep_dict)
                else:
                    # Record the deployment. The CAS no-ops only if a /cancel raced finalization
                    # — then the adapter we just registered is orphaned, so deregister it and
                    # report the conflict instead of a bogus 200.
                    marked = mark_deployed(run_id, dep_dict, expect_state=prev_state)
                    if marked.state != "deployed":
                        with contextlib.suppress(Exception):
                            undeploy_adapter(run_id)
                        raise HTTPException(
                            status_code=409,
                            detail=f"run {run_id} became {marked.state!r} during deploy; aborted",
                        )
            return dep_dict

    @app.get("/v1/runs/{run_id}/checkpoints")
    def run_checkpoints(run_id: str, key: dict = Depends(require_key)):
        """List a run's deployable per-step RL checkpoints (each `flash deploy --step N`-able).

        Reads the snapshots the worker streamed to HF, and best-effort mirrors them to the
        backend store so a listing also persists them."""
        status = owned_run(run_id, key)
        spec = JobSpec.from_dict(status.spec)
        checkpoints = list_checkpoints(spec)
        with contextlib.suppress(Exception):
            from flash.server.checkpoints import register_checkpoints_best_effort

            register_checkpoints_best_effort(status)
        return {"run_id": run_id, "checkpoints": checkpoints}

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
            # Delete is idempotent: a missing serving-side adapter still means the local
            # deployment record can be cleared.
            if status.deployment:
                mark_undeployed(run_id)
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
            if payload.get("stream") is True:
                return StreamingResponse(
                    serve_chat_stream(
                        run_id=run_id,
                        messages=payload.get("messages") or [],
                        temperature=float(payload.get("temperature") or 0.0),
                        max_tokens=int(payload.get("max_tokens") or 512),
                        # a run trained with thinking serves with thinking (per-run parity)
                        thinking=spec.thinking,
                    ),
                    media_type="text/plain; charset=utf-8",
                )
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
