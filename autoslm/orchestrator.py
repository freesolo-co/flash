"""Platform orchestrator: drives managed RunPod Flash GPUs (one per run)."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

from .catalog import ModelInfo, resolve_model
from .worker_spec import JobSpec

RUNS_DIR = os.environ.get("AUTOSLM_RUNS_DIR", ".autoslm/runs")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
TERMINAL_STATES = frozenset({"done", "failed", "cancelled", "dry_run"})
# Serializes the read-check-write in _update so a status transition is an atomic
# compare-and-set (the control plane is single-instance with per-run threads).
_STATUS_LOCK = threading.Lock()


def artifacts_dir(spec: JobSpec) -> str:
    """Run-scoped artifact root: results/runpod/<phase>/<run_id>."""
    return os.path.join(RESULTS_DIR, "runpod", spec.phase, spec.run_id)


def adapter_prefix(spec: JobSpec, seed: int | None = None) -> str:
    """A run's adapter location on the HF artifact store."""
    chosen = spec.train.seeds[0] if seed is None else seed
    return f"{spec.phase}/{spec.run_id}/seed{chosen}"


def _gpu_rate(gpu_type: str) -> float:
    """Representative $/hr for cost projection (live RunPod pricing, static fallback);
    the worker also records wall time so cost = wall_hours * rate."""
    try:
        from autoslm.flash.pricing import hourly_rate

        return hourly_rate(gpu_type)
    except Exception:
        return 0.80


@dataclass
class RunStatus:
    run_id: str
    state: str
    spec: dict
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cost_usd: float = 0.0
    error: str | None = None
    artifacts_dir: str | None = None
    deployment: dict | None = None
    # Durable job handle {endpoint_id, endpoint_name, job_id} — lets any process
    # reattach to / cancel the remote job (see `slm attach`).
    remote: dict | None = None
    # Index of the next seed to run for a multi-seed job, set while the remote handle
    # is cleared in the gap between seeds. Lets recover_runs resume the remaining seeds
    # after an inter-seed restart instead of failing the run (losing completed work).
    resume_seed_index: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class _RunCancelled(RuntimeError):
    """User cancellation observed mid-run; terminal, never retried/overwritten."""


def new_run_id(prefix: str = "autoslm") -> str:
    return f"{prefix}-{int(time.time())}-{uuid.uuid4().hex[:8]}"


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def require_safe_run_id(run_id: str) -> str:
    """Reject run ids that could traverse outside the runs directory.

    Run ids flow from API path params into filesystem paths (status json,
    log files); restrict them to a conservative filename alphabet.
    """
    if not _RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return run_id


def runs_file_path(run_id: str, suffix: str) -> str:
    """Containment-checked path for a run's file under RUNS_DIR.

    Belt and braces with require_safe_run_id: the resolved path must stay
    inside the runs directory even if the alphabet check ever regresses.
    """
    base = os.path.abspath(RUNS_DIR)
    path = os.path.normpath(os.path.join(base, f"{require_safe_run_id(run_id)}{suffix}"))
    if not path.startswith(base + os.sep):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return path


def _with_model_disk(spec: JobSpec, info: ModelInfo) -> dict:
    """Spec dict with gpu.disk_gb raised to the model's min_disk_gb (catalog).

    Big-checkpoint models (e.g. the 35B-A3B MoE at ~72 GB bf16) need more container
    disk than the platform's 64 GB default; this makes them work without users having
    to know the right ``gpu.disk_gb``.
    """
    d = spec.to_dict()
    need = int(getattr(info, "min_disk_gb", 0) or 0)
    if need > int(d["gpu"].get("disk_gb") or 0):
        d["gpu"] = {**d["gpu"], "disk_gb": need}
    return d


def submit_job(spec: JobSpec, dry_run: bool = False, background: bool = False) -> RunStatus:
    """Submit a job. In real mode this provisions a RunPod Flash GPU; dry-run only records state."""
    info = resolve_model(spec.model, spec.algorithm, policy=spec.model_policy, gpu=spec.gpu.type)
    spec = JobSpec.from_dict(
        {**_with_model_disk(spec, info), "run_id": spec.run_id or new_run_id()}
    )
    status = RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
    _save_status(status)
    if dry_run:
        status.state = "dry_run"
        _save_status(status)
        return status
    if background:
        threading.Thread(target=_run_job, args=(spec,), daemon=True).start()
        return get_status(spec.run_id)
    _run_job(spec)
    return get_status(spec.run_id)


def get_status(run_id: str) -> RunStatus:
    path = runs_file_path(run_id, ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"unknown run_id: {run_id}")
    with open(path) as f:
        return RunStatus(**json.load(f))


def list_runs() -> list[RunStatus]:
    os.makedirs(RUNS_DIR, exist_ok=True)
    runs = []
    for name in sorted(os.listdir(RUNS_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(RUNS_DIR, name)) as f:
                runs.append(RunStatus(**json.load(f)))
    return runs


def get_logs(run_id: str) -> str:
    log_path = runs_file_path(run_id, ".log")
    if not os.path.exists(log_path):
        return ""
    with open(log_path) as f:
        return f.read()


def cancel_run(run_id: str) -> RunStatus:
    """Cancel a run: delete its remote Flash endpoint (stopping the worker), then mark it
    cancelled.

    Uses ``terminate_endpoint`` (reconstructs the run's uniquely-named endpoint and deletes it
    via the RunPod API) so the cancel works **cross-process** — a fresh ``slm cancel`` actually
    stops the GPU worker, instead of leaving it running until the wall cap. Best-effort: any
    teardown error is recorded but still flips the run to ``cancelled``.
    """
    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    spec = JobSpec.from_dict(status.spec)
    remote = status.remote or {}
    # A deployed run also owns a serving endpoint (autoslm-serve-*) that the
    # training-endpoint GC below does not touch; tear it down too so a
    # cancelled run can't leave a billable deployment registered. Serving is
    # RunPod-only, so use the class actually deployed (a Vast-only training class
    # falls back to a RunPod class at deploy time).
    if status.state == "deployed":
        try:
            from autoslm.serve.deploy import undeploy_adapter

            deployed_gpu = (status.deployment or {}).get("gpu") or spec.gpu.type
            deleted = undeploy_adapter(run_id, gpu_name=deployed_gpu)
            # Mark the deployment inactive so /v1/deployments and /chat (which gate only
            # on the deployment record's state) stop treating the cancelled run as
            # active. dev mode is scale-to-zero: a never-chatted dev deployment has no
            # endpoint yet, so an empty deletion is still a clean teardown — don't leave
            # it "ready". always-on provisions at deploy time, so only mark it inactive
            # once a deletion is confirmed (an empty deletion there is suspicious).
            dev_mode = (status.deployment or {}).get("mode", "dev") == "dev"
            if status.deployment and (deleted or dev_mode):
                status.deployment = {**status.deployment, "state": "undeployed"}
                _save_status(status)
        except Exception:
            # Best-effort serving teardown: a failure here must not block the cancel
            # below (the run still flips to cancelled and the training endpoint is GC'd).
            pass
    # Durable path first: stop the exact remote worker via the provider's REST API
    # (works from any process); endpoint/instance teardown is shared with the GC.
    if remote.get("provider") == "vast":
        try:
            from autoslm.providers import vast as vast_provider

            vast_provider.cancel(remote)
            # Belt and suspenders: catch any instance of this run the handle missed.
            vast_provider.destroy_run_instances(run_id)
        except Exception:
            # Best-effort remote stop; the orphan sweep / endpoint GC are the backstop.
            pass
    elif status.remote:
        try:
            from autoslm.flash import runpod_api

            runpod_api.cancel_job(status.remote["endpoint_id"], status.remote["job_id"])
        except Exception:
            # Best-effort remote stop; _gc_run_endpoints below still tears the endpoint down.
            pass
    # RunPod endpoint GC only — the vast branch above already destroyed its instance
    # (calling terminate_endpoint for a vast run would be a no-op against RunPod).
    if remote.get("provider") != "vast":
        _gc_run_endpoints(spec)
    _update(run_id, "cancelled")
    return get_status(run_id)


def attach_run(run_id: str, log_stream=None) -> RunStatus:
    """Re-attach to a run's remote job from ANY process (after a client crash/restart).

    Uses the persisted {endpoint_id, job_id} handle to resume polling; on completion,
    persists metrics exactly like the original client would have, flips the state, and
    GCs the endpoint. Raises if the run has no persisted handle (it failed or was
    cancelled before a worker was provisioned).
    """
    import sys

    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    if not status.remote:
        raise ValueError(f"run {run_id} has no persisted job handle; cannot reattach")
    from autoslm.flash.durable import make_hf_heartbeat_reader

    spec = JobSpec.from_dict(status.spec)
    remote = dict(status.remote)
    seed = int(remote.pop("seed", spec.train.seeds[0]))
    log = log_stream or sys.stderr
    hf_repo = os.environ.get("HF_REPO", "")
    reader = make_hf_heartbeat_reader(hf_repo, adapter_prefix(spec, seed)) if hf_repo else None
    if remote.get("provider") == "vast":
        from autoslm.providers.vast import VastJobHandle, poll_vast_job

        vhandle = VastJobHandle.from_dict(remote)
        print(f"attaching to {run_id}: vast instance={vhandle.instance_id}", file=log)
        res = poll_vast_job(vhandle, spec, seed, log=log, heartbeat_reader=reader)
    else:
        from autoslm.flash.durable import JobHandle, poll_job

        handle = JobHandle.from_dict(remote)
        print(
            f"attaching to {run_id}: job={handle.job_id} endpoint={handle.endpoint_name}", file=log
        )
        res = poll_job(handle, log=log, heartbeat_reader=reader)
    try:
        # A best-effort cancel deletes the job/instance, which the poller reports as a
        # failure (or a late worker may still succeed) — either way, re-read the state
        # first so a recovery thread can't overwrite the user's terminal `cancelled`.
        if get_status(run_id).state == "cancelled":
            return get_status(run_id)
        if not res.ok:
            _update(run_id, "failed", error=f"{res.failure}: {res.detail}")
            return get_status(run_id)
        # Earlier seeds of a multi-seed run already persisted their cost into
        # status.cost_usd; add this seed's so recovery doesn't underreport spend.
        total = float(status.cost_usd or 0.0) + _persist_metrics(spec, seed, res.metrics)
        # A cancel can land while this thread persists the recovered seed's metrics
        # (after the late-cancel check above). Re-read before the post-seed writes so
        # the "running" update and the terminal "done" below can't resurrect a
        # user-cancelled run (mirrors the fresh seed loop). _RunCancelled is caught
        # below, leaving the cancellation intact.
        if get_status(run_id).state == "cancelled":
            raise _RunCancelled(f"run {run_id} was cancelled")
        # The remote handle only identifies the seed that was in flight. For a
        # multi-seed run, resume the remaining seeds instead of terminally
        # completing the whole run after just this one.
        try:
            resumed_index = list(spec.train.seeds).index(seed) + 1
        except ValueError:
            resumed_index = len(spec.train.seeds)
        more_seeds = resumed_index < len(spec.train.seeds)
        # Clear the now-stale completed handle before resuming. In the
        # allocation/provisioning gap before the next seed's on_handle() persists a
        # fresh handle, a server restart must not reattach recovery to this finished
        # job — that would double-count its cost and replay the wrong seed. Record the
        # next seed index so a restart in that gap resumes the remaining seeds rather
        # than failing the run. (The last seed keeps its handle for post-run
        # observability, mirroring the fresh-submit seed loop.)
        _update(
            run_id,
            "running",
            cost_usd=total,
            artifacts_dir=artifacts_dir(spec),
            **({"remote": None, "resume_seed_index": resumed_index} if more_seeds else {}),
        )
        if more_seeds:
            _run_seed_loop(spec, log, start_index=resumed_index, prior_cost=total)
        else:
            _update(run_id, "done", cost_usd=total, artifacts_dir=artifacts_dir(spec))
    except _RunCancelled:
        # Intentional: cancel_run already wrote the terminal `cancelled` state; leave it.
        pass
    except Exception as exc:
        if get_status(run_id).state != "cancelled":
            _update(run_id, "failed", error=str(exc))
    finally:
        if remote.get("provider") == "vast":
            # Vast bills until destroyed: kill the exact instance this handle owns.
            try:
                from autoslm.providers import vast_api

                vast_api.destroy_instance(int(remote["instance_id"]))
            except Exception:
                # Best-effort teardown; the vast orphan sweep reaps anything left behind.
                pass
        else:
            _gc_run_endpoints(spec)
    return get_status(run_id)


def resume_run(run_id: str, log_stream=None) -> RunStatus:
    """Resume the remaining seeds of a multi-seed run after a restart in the inter-seed gap.

    Between two seeds the completed seed's handle is cleared and ``resume_seed_index`` is
    recorded (see ``_run_seed_loop``). A control-plane restart in that handle-less window
    must RESUME from that index rather than fail the run and discard the finished seeds.
    Unlike ``attach_run`` there is no live job to poll — the prior process already tore the
    seed's endpoint down — so we start a fresh seed loop from the recorded index. The slm
    package was uploaded to HF on the original submit, so the worker can still fetch it; no
    re-upload is needed.
    """
    import sys

    status = get_status(run_id)
    if status.state in TERMINAL_STATES:
        return status
    if status.resume_seed_index is None:
        raise ValueError(f"run {run_id} has no resume_seed_index; cannot resume")
    spec = JobSpec.from_dict(status.spec)
    log = log_stream or sys.stderr
    print(f"resuming {run_id}: remaining seeds from index {status.resume_seed_index}", file=log)
    try:
        _run_seed_loop(
            spec,
            log,
            start_index=status.resume_seed_index,
            prior_cost=float(status.cost_usd or 0.0),
        )
    except _RunCancelled:
        pass  # cancel_run already set the terminal state
    except Exception as exc:
        if get_status(run_id).state != "cancelled":
            _update(run_id, "failed", error=str(exc))
    finally:
        # Mirror _run_job: the resume path also marked this run active in recover_runs, so
        # the startup orphan sweep skipped its instances — GC any endpoint a transient
        # destroy left behind rather than leaking a billable Vast instance.
        _gc_run_endpoints(spec)
    return get_status(run_id)


def mark_deployed(run_id: str, deployment: dict) -> RunStatus:
    # Atomic + terminal-respecting (same guard as _update): a /cancel landing during
    # always-on provisioning/warmup writes `cancelled`; this must NOT overwrite it with
    # `deployed` and resurrect the run as an active deployment.
    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state in TERMINAL_STATES:
            return status
        status.deployment = deployment
        status.state = "deployed"
        _save_status(status)
        return status


def rollback_deploy(run_id: str, snapshot: RunStatus) -> None:
    """Restore the pre-deploy state/deployment after always-on provisioning fails.

    Lock-guarded + terminal-respecting (same guard as _update/mark_deployed): a /cancel
    that landed during provisioning/warmup already persisted `cancelled`; restoring the
    pre-deploy snapshot must NOT overwrite it and resurrect the run as `done`/`deployed`.
    """
    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state in TERMINAL_STATES:
            return
        status.state = snapshot.state
        status.deployment = snapshot.deployment
        status.updated_at = time.time()
        _save_status(status)


def _run_job(spec: JobSpec) -> None:
    # Lazy import so dry-run / unit tests never construct a Flash endpoint.
    from autoslm.flash.train import submit_train, upload_code

    # A cancel can land between the queued status being returned to the client and
    # this background thread starting; don't overwrite a terminal state (cancelled)
    # with provisioning and then launch a paid seed as if the cancel never happened.
    if get_status(spec.run_id).state in TERMINAL_STATES:
        return
    _update(spec.run_id, "provisioning")
    log_path = os.path.join(RUNS_DIR, f"{spec.run_id}.log")
    try:
        _run_job_inner(spec, log_path, submit_train, upload_code)
    finally:
        # Endpoint GC: every run leaves its uniquely-named endpoint registered, and the
        # account-wide *max workers quota* (5 by default) counts registered endpoints —
        # after a handful of runs, ALL new submissions fail with "Max workers across all
        # endpoints must not exceed your workers quota". Tear ours down on any terminal
        # state (best-effort; never raises).
        _gc_run_endpoints(spec)


def _spec_with_gpu(spec: JobSpec, gpu_type: str) -> JobSpec:
    """The spec the workers/loggers see for THIS attempt's allocated class."""
    if spec.gpu.type == gpu_type:
        return spec
    d = spec.to_dict()
    d["gpu"] = {**d["gpu"], "type": gpu_type}
    return JobSpec.from_dict(d)


def _submit_seed_supervised(spec: JobSpec, seed: int, log) -> dict:
    """Run one seed with the durable submit/poll path + bounded auto-retry.

    Each attempt first ALLOCATES the GPU: the cheapest class across providers (RunPod
    live pricing + Vast verified-datacenter offers) that fits the model — re-resolved
    fresh per attempt because offers are a live market. A policy ``gpu.requested``
    ("cheapest"/"auto") lets the allocator pick the class; a concrete ``gpu.requested``
    pins the class (the allocator then only picks the provider); ``gpu.provider`` pins
    the substrate.

    Retries (fresh job on a fresh host; worker resumes from the latest HF
    checkpoint) when the failure looks infra-shaped: a stall (heartbeat frozen), a
    client polling breakdown, or a platform TIMED_OUT/worker-loss. Sick Vast
    machines are blacklisted for the run; failover naturally crosses providers.
    Genuine worker errors (the run's code crashed; traceback persisted to HF) fail
    immediately. The offline test/CI marker AUTOSLM_SKIP_NET takes the blocking
    in-process submit instead (the durable poll path is network-only).
    """
    import autoslm.flash.durable as durable
    from autoslm.flash.train import submit_train

    if os.environ.get("AUTOSLM_SKIP_NET"):
        return submit_train(spec, seed, log=log)

    from autoslm.flash.gpus import POLICY_NAMES
    from autoslm.providers import vast as vast_provider
    from autoslm.providers.allocator import allocate, allocation_summary

    last_handle: dict = {}
    # Every RunPod endpoint id this run registered across attempts. Retries run on
    # rN-suffixed endpoints whose names _gc_run_endpoints cannot reconstruct, and a
    # failed delete during the next attempt's teardown would otherwise lose the id;
    # GC the whole set at exit so no retry endpoint leaks against the worker quota.
    seen_endpoints: set[str] = set()

    def on_handle(handle: dict):
        last_handle.clear()
        last_handle.update(handle)
        if handle.get("endpoint_id"):
            seen_endpoints.add(handle["endpoint_id"])
        _update(spec.run_id, "running", remote={**handle, "seed": int(seed)})

    def _gc_seen_endpoints() -> None:
        if not seen_endpoints:
            return
        from autoslm.flash import runpod_api

        for eid in seen_endpoints:
            with contextlib.suppress(Exception):
                runpod_api.delete_endpoint(eid)

    max_retries = int(spec.gpu.max_retries)
    last_detail = None
    bad_machines: set[int] = set()
    # Re-allocate freely for policy requests ("cheapest"/"auto"); honor a concrete
    # user pin by passing it through as the only candidate class.
    requested = (spec.gpu.requested or "").strip().lower()
    pinned_gpu = None if requested in POLICY_NAMES else spec.gpu.type
    for attempt in range(max_retries + 1):
        if attempt > 0 and last_handle:
            # A stalled/timed-out attempt often means the worker is pinned to a
            # throttled/sick host; tear it down so the fresh deploy lands elsewhere.
            if last_handle.get("provider") == "vast":
                from autoslm.providers import vast_api

                vast_api.destroy_instance(int(last_handle.get("instance_id") or 0))
                if last_handle.get("machine_id"):
                    bad_machines.add(int(last_handle["machine_id"]))
                print(
                    f"retry {attempt}: destroyed vast instance "
                    f"{last_handle.get('instance_id')} (machine "
                    f"{last_handle.get('machine_id')} blacklisted for this run)",
                    file=log,
                    flush=True,
                )
            elif last_handle.get("endpoint_id"):
                try:
                    from autoslm.flash import runpod_api

                    runpod_api.cancel_job(last_handle["endpoint_id"], last_handle["job_id"])
                    runpod_api.delete_endpoint(last_handle["endpoint_id"])
                    print(
                        f"retry {attempt}: deleted endpoint {last_handle['endpoint_id']} "
                        "(escaping throttled/sick host)",
                        file=log,
                        flush=True,
                    )
                except Exception:
                    # Logging the host-escape note is cosmetic; never let it abort the retry.
                    pass
            # The previous endpoint is now deleted; clear the persisted handle so a cancel
            # or control-plane restart during the fresh deploy doesn't operate on (or get
            # shielded by) the dead handle. The next on_handle() records the new one.
            with contextlib.suppress(FileNotFoundError):
                st = get_status(spec.run_id)
                if st.state not in TERMINAL_STATES and st.remote is not None:
                    _update(spec.run_id, st.state, remote=None)
        res = None
        alloc = None
        # A cancel can land after _run_seed_loop's pre-submit check but while
        # allocation/pricing runs, when no handle exists yet for cancel_run() to
        # delete. Re-read state right before paid provisioning so a cancelled run
        # never launches a worker (the later checks only stop the final-state
        # overwrite, after the GPU has already run and billed).
        with contextlib.suppress(FileNotFoundError):
            if get_status(spec.run_id).state == "cancelled":
                raise _RunCancelled(f"run {spec.run_id} was cancelled")
        try:
            alloc = allocate(
                spec.model,
                spec.algorithm,
                gpu=pinned_gpu,
                provider=spec.gpu.provider,
                disk_gb=spec.gpu.disk_gb,
                allow_unvalidated=spec.gpu.allow_unvalidated,
                exclude_machine_ids=frozenset(bad_machines),
            )
        except Exception as exc:
            from autoslm.flash.gpus import UnsupportedGpuError

            if isinstance(exc, UnsupportedGpuError):
                raise  # config-shaped: no GPU anywhere can run this job
            res = durable.PollResult(False, failure="poll_error", detail=f"allocation: {exc}")
        if alloc is not None:
            # allocate() above ran a live-market price walk; re-check cancellation
            # right before provisioning so a cancel during allocation doesn't still
            # launch a paid worker.
            with contextlib.suppress(FileNotFoundError):
                if get_status(spec.run_id).state == "cancelled":
                    raise _RunCancelled(f"run {spec.run_id} was cancelled")
            print(allocation_summary(alloc), file=log, flush=True)
            run_spec = _spec_with_gpu(spec, alloc.gpu)
            try:
                if alloc.provider == "vast":
                    # Offer book for the live-market walk: the chosen class first,
                    # then other allocator-approved classes by price.
                    ok_classes = {c.gpu for c in alloc.candidates if c.provider == "vast"}
                    offers = sorted(
                        (o for o in alloc.vast_offers if o.gpu in ok_classes),
                        key=lambda o: (o.gpu != alloc.gpu, o.dph_total),
                    )
                    res = vast_provider.submit_train_durable_vast(
                        run_spec,
                        seed,
                        log=log,
                        on_handle=on_handle,
                        attempt=attempt,
                        offers=offers,
                    )
                else:
                    res = durable.submit_train_durable(
                        run_spec, seed, log=log, on_handle=on_handle, attempt=attempt
                    )
            except Exception as exc:
                # Deploy/submit themselves can fail transiently (observed: RunPod
                # GraphQL "Something went wrong" x3 during a retry deploy; a vast
                # offer pool emptying between search and rent). That must consume a
                # retry, not kill the run — the budget exists precisely for flakes.
                res = durable.PollResult(
                    False, failure="poll_error", detail=f"deploy/submit: {exc}"
                )
                if attempt < max_retries:
                    time.sleep(10 * (attempt + 1))  # let the transient clear
        if res.ok:
            # A best-effort cancel may fail to stop the worker, which then completes
            # successfully after cancel_run() persisted `cancelled`. Don't let a late
            # worker success resurrect the run into running/done.
            try:
                if get_status(spec.run_id).state == "cancelled":
                    raise _RunCancelled(f"run {spec.run_id} was cancelled")
            except FileNotFoundError:
                # Status file not yet written (early race): treat as not-cancelled, proceed.
                pass
            # Worker is done (DONE sentinel seen); GC every endpoint this seed used,
            # including intermediate rN retries _gc_run_endpoints can't name.
            _gc_seen_endpoints()
            # Record the class actually allocated so _persist_metrics rates the right
            # RunPod card when a policy GPU was re-allocated away from the provisional.
            if alloc is not None and isinstance(res.metrics, dict):
                res.metrics.setdefault("allocated_gpu", alloc.gpu)
            return res.metrics
        last_detail = f"{res.failure}: {res.detail}"
        # Infra-shaped failures are retried on a FRESH endpoint/host; genuine worker
        # code errors are not. Detail markers cover the observed flake classes:
        # platform timeouts, worker pip-install network timeouts, and sick-GPU hosts.
        _infra_markers = (
            "TIMED_OUT",
            "Failed to fetch",
            "operation timed out",
            "python_dependencies failed",
            "Connection reset",
            "cuda not available",
            "GPU never became ready",
        )
        infra_shaped = res.failure in ("stalled", "poll_error") or any(
            m in (res.detail or "") for m in _infra_markers
        )
        # A cancel deletes the endpoint, which the poller sees as an
        # infra-shaped failure; retrying would resurrect the run and keep
        # billing. The user's cancel wins over the retry budget.
        try:
            if get_status(spec.run_id).state == "cancelled":
                raise _RunCancelled(f"run {spec.run_id} was cancelled")
        except FileNotFoundError:
            # Status file not yet written (early race): treat as not-cancelled and proceed.
            pass
        print(
            f"seed={seed} attempt={attempt} failed ({res.failure}); "
            f"{'retrying (resume from last checkpoint)' if infra_shaped and attempt < max_retries else 'not retrying'}"
            f"\n--- failure detail ---\n{(res.detail or '')[:2000]}\n---",
            file=log,
            flush=True,
        )
        if not infra_shaped or attempt >= max_retries:
            break
    # Retry budget exhausted: GC every endpoint this seed registered (the final
    # attempt's is in status.remote for _gc_run_endpoints, but intermediate rN ones
    # are only known here).
    _gc_seen_endpoints()
    raise RuntimeError(f"seed {seed} failed after retries: {last_detail}")


def _run_job_inner(spec: JobSpec, log_path: str, submit_train, upload_code) -> None:
    try:
        upload_code()  # ship the slm package to HF so the GPU worker can run it
        with open(log_path, "a") as log:
            _run_seed_loop(spec, log, start_index=0, prior_cost=0.0)
    except _RunCancelled:
        return  # cancel_run already set the terminal state
    except Exception as exc:
        if get_status(spec.run_id).state != "cancelled":
            _update(spec.run_id, "failed", error=str(exc))
        raise


def _run_seed_loop(spec: JobSpec, log, *, start_index: int, prior_cost: float) -> None:
    """Run spec.train.seeds[start_index:] under supervision; finalize the run.

    Shared by a fresh submit (start_index=0) and post-restart recovery, which
    resumes the remaining seeds after the in-flight one completes."""
    total_cost = prior_cost
    seeds = spec.train.seeds
    for i in range(start_index, len(seeds)):
        seed = seeds[i]
        # An early cancel (before any remote handle existed) sets `cancelled`;
        # do not overwrite it with `running` and submit the GPU job anyway.
        if get_status(spec.run_id).state == "cancelled":
            raise _RunCancelled(f"run {spec.run_id} was cancelled")
        _update(spec.run_id, "running")
        print(
            f"starting seed={seed} phase={spec.phase} model={spec.model} gpu={spec.gpu.type}",
            file=log,
            flush=True,
        )
        metrics = _submit_seed_supervised(spec, seed, log)
        total_cost += _persist_metrics(spec, seed, metrics)
        # A cancel can land while this thread writes metrics — after the supervised
        # late-cancel check. Re-read before the post-seed status writes so a late
        # worker success doesn't resurrect a user-cancelled run via this "running"
        # update (or the final "done" below).
        with contextlib.suppress(FileNotFoundError):
            if get_status(spec.run_id).state == "cancelled":
                raise _RunCancelled(f"run {spec.run_id} was cancelled")
        # If more seeds follow, this seed's endpoint/instance is already torn down, so
        # clear the now-stale remote handle: a restart in the gap before the next
        # seed's on_handle must not make recover_runs reattach to a deleted handle and
        # fail the run. Record the next seed index so a restart in that handle-less gap
        # RESUMES the remaining seeds (recover_runs) instead of discarding the completed
        # ones. The last seed keeps its handle for post-run observability (the run is
        # about to go terminal, which recover_runs never reattaches).
        more_seeds = (i + 1) < len(seeds)
        _update(
            spec.run_id,
            "running",
            cost_usd=total_cost,
            **({"remote": None, "resume_seed_index": i + 1} if more_seeds else {}),
        )
        print(
            f"seed={seed} done: train_wall={metrics.get('wall_seconds')} cost_usd={total_cost:.4f}",
            file=log,
            flush=True,
        )
    # Final guard: a cancel landing after the last seed's check must not be overwritten
    # by the terminal "done".
    with contextlib.suppress(FileNotFoundError):
        if get_status(spec.run_id).state == "cancelled":
            raise _RunCancelled(f"run {spec.run_id} was cancelled")
    _update(
        spec.run_id,
        "done",
        cost_usd=total_cost,
        artifacts_dir=artifacts_dir(spec),
        resume_seed_index=None,
    )


def _gc_run_endpoints(spec: JobSpec) -> None:
    """Best-effort teardown of every endpoint a run may have registered.

    Retried attempts run on rN-suffixed endpoints whose runpod_flash state is
    isolated per-suffix, so the name-based terminate_endpoint cannot see them;
    the persisted remote handle's endpoint id covers whichever attempt ran
    last via the plain REST API."""
    status = None
    with contextlib.suppress(Exception):
        status = get_status(spec.run_id)
    if status is not None and status.remote:
        try:
            from autoslm.flash import runpod_api

            runpod_api.delete_endpoint(status.remote["endpoint_id"])
        except Exception:
            # Best-effort GC; the name-reconstructed terminate_endpoint below is the backstop.
            pass
    try:
        from autoslm.flash.train import terminate_endpoint

        terminate_endpoint(spec.gpu.type, spec.run_id)
    except Exception:
        # Best-effort GC; an undeleted endpoint only holds worker quota, never blocks the run.
        pass
    # Vast instances bill until destroyed: the runner's per-attempt `finally` already
    # destroys them, but a crashed supervisor thread can leave one behind. Destroy any
    # instance still labeled for this run (best-effort; never raises).
    if os.environ.get("VAST_API_KEY"):
        try:
            from autoslm.providers import vast as vast_provider

            vast_provider.destroy_run_instances(spec.run_id)
        except Exception:
            # Best-effort orphan sweep; a leftover vast instance is caught by the
            # periodic sweep_orphans pass, so don't let teardown failure raise here.
            pass


def _persist_metrics(spec: JobSpec, seed: int, metrics: dict) -> float:
    """Write metrics to results/runpod/<phase>/<run_id>/seedN and return the cost.

    The run id keeps concurrent/sequential runs of the same phase+seed from
    overwriting each other's artifacts. Vast runs arrive with ``cost_usd`` already
    stamped from the offer's real $/hr (plus provider notes) and short-circuit the
    rate fallback below (the RunPod projection)."""
    dest = os.path.join(artifacts_dir(spec), f"seed{seed}")
    os.makedirs(dest, exist_ok=True)
    # Rate the actually-allocated class, not the parse-time provisional spec.gpu.type:
    # a policy GPU can be re-allocated to a different RunPod class at submit time, so
    # the worker stamps "allocated_gpu" into metrics for the cost fallback below.
    gpu_type = metrics.get("allocated_gpu") or spec.gpu.type
    rate = _gpu_rate(gpu_type)
    cost = metrics.get("cost_usd") or 0.0
    if not cost:
        wall = float(metrics.get("wall_seconds") or 0.0)
        cost = wall / 3600.0 * rate
        metrics = {**metrics, "cost_usd": cost}
        metrics.setdefault("notes", {})
        if isinstance(metrics["notes"], dict):
            metrics["notes"]["provider"] = "runpod"
            metrics["notes"]["runpod_rate_usd_hr"] = rate
            metrics["notes"]["runpod_gpu"] = gpu_type
    with open(os.path.join(dest, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return float(cost)


def _update(run_id: str, state: str, **updates) -> None:
    # The read-check-write below must be atomic: a concurrent `slm cancel` (also via
    # _update) landing between the get_status read and the _save_status write could
    # otherwise be clobbered by this stale background update, resurrecting a cancelled
    # run. The control plane is single-instance with per-run threads, so a process-wide
    # lock serializes all status transitions into a compare-and-set.
    with _STATUS_LOCK:
        status = get_status(run_id)
        # Terminal states are STICKY: once a run is done/failed/cancelled/dry_run, no
        # other state may overwrite it. This closes the whole cancel-race class at the
        # source — a cancel landing between a caller's check and a later write
        # (provisioning/running, or even a late terminal done/failed from a worker that
        # finished as the cancel arrived) can no longer resurrect the run. Same-state
        # writes still pass so terminal field updates (cost_usd, error, artifacts_dir)
        # are preserved.
        if status.state in TERMINAL_STATES and state != status.state:
            return
        status.state = state
        status.updated_at = time.time()
        for key, value in updates.items():
            setattr(status, key, value)
        _save_status(status)


def _save_status(status: RunStatus) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    # Write-then-rename: a concurrent reader (poll on /v1/runs or /logs) must
    # never observe a half-written/truncated file and 500 on JSONDecodeError.
    # The temp name is UNIQUE per write (mkstemp) so two threads updating the same
    # run (e.g. a cancel racing the background seed update) can't clobber each
    # other's temp file mid-dump — each os.replace is atomic and independent.
    path = runs_file_path(status.run_id, ".json")
    fd, tmp = tempfile.mkstemp(dir=RUNS_DIR, prefix=f"{status.run_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(status.to_dict(), f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
