"""Platform runner: drives managed GPUs across providers (RunPod Flash + Vast), one allocation per seed."""

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
from .spec import JobSpec

# Fixed local storage roots (not operator-configurable): run-state JSON + result artifacts,
# both under the ~/.flash state dir (same root as server/db.py's DB_PATH) so a single
# directory holds all control-plane state — mount one volume at ~/.flash to persist it.
# Tests redirect them via monkeypatch.setattr(runner, "RUNS_DIR"/"RESULTS_DIR").
_STATE_DIR = os.path.join(os.path.expanduser("~"), ".flash")
RUNS_DIR = os.path.join(_STATE_DIR, "runs")
RESULTS_DIR = os.path.join(_STATE_DIR, "results")
TERMINAL_STATES = frozenset({"done", "failed", "cancelled", "dry_run"})
# Terminal states a deploy must NOT overwrite. `done` is terminal but IS deployable
# (deploying a finished run is the whole point), so it's excluded here; cancelled/failed/
# dry_run must never be flipped to `deployed`.
_UNDEPLOYABLE_STATES = TERMINAL_STATES - {"done"}
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
        from flash.providers.runpod.pricing import hourly_rate

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


def new_run_id(prefix: str = "flash") -> str:
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

    Big-checkpoint models (whose weights alone exceed the default) need more container
    disk than the platform's 64 GB default; this makes them work without users having
    to know the right ``gpu.disk_gb``.
    """
    d = spec.to_dict()
    need = int(getattr(info, "min_disk_gb", 0) or 0)
    if need > int(d["gpu"].get("disk_gb") or 0):
        d["gpu"] = {**d["gpu"], "disk_gb": need}
    return d


def submit_job(spec: JobSpec, dry_run: bool = False, background: bool = False) -> RunStatus:
    """Submit a job. In real mode this allocates and provisions the cheapest validated GPU class
    across the configured providers (RunPod Flash or Vast); dry-run only records state."""
    info = resolve_model(
        spec.model, spec.algorithm, policy=spec.model_policy, gpu=spec.gpu.type, train=spec.train
    )
    # Fail fast: a disaggregated-only model (e.g. Qwen3.6-35B-A3B) can't run colocated GRPO, and a
    # single-trainer-only model (the 35B) can't use a multi-trainer (>1 trainer card) DDP split.
    from .engine.rollout_bench import validate_disaggregated_requirement

    validate_disaggregated_requirement(
        requires_disaggregated=info.requires_disaggregated,
        algorithm=spec.algorithm,
        inference_gpus=spec.train.inference_gpus,
        single_trainer_only=getattr(info, "single_trainer_only", False),
        gpu_count=spec.gpu.count,
    )
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
    # Whether the run was a live `deployed` serving run at cancel entry. This scopes the
    # final `cancelled` transition's terminal override below: only a `deployed` run can have
    # a concurrent undeploy (`mark_undeployed`) race this teardown and write a non-completion
    # terminal `done`. A non-deployed run (running/provisioning/queued) has an in-flight
    # TRAINING thread whose only terminal `done` is a GENUINE completion — which cancel must
    # never clobber. See the final _update call for how this gates the override.
    entered_deployed = status.state == "deployed"
    spec = JobSpec.from_dict(status.spec)
    remote = status.remote or {}
    # A deployed run also owns a serving endpoint (flash-serve-*) that the
    # training-endpoint GC below does not touch; tear it down too so a
    # cancelled run can't leave a billable deployment registered. Serving is
    # RunPod-only, so use the class actually deployed (a Vast-only training class
    # falls back to a RunPod class at deploy time).
    if status.state == "deployed":
        try:
            from flash.serve.deploy import undeploy_adapter

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
                # Mark the deployment inactive through the lock-guarded path so this write
                # participates in the same _STATUS_LOCK as the rest of the runner. A bare
                # _save_status here would persist a stale pre-teardown snapshot OUTSIDE the
                # lock, bypassing serialization and potentially clobbering a concurrent field
                # update. We mark ONLY the deployment field and leave the run's state alone
                # (no state re-assert): a concurrent mark_undeployed can move the run to
                # terminal `done` between our get_status read and this write, and _update's
                # compare-and-set rejects ANY transition off a terminal state (even a
                # same-field re-assert of the stale `deployed`), which would silently leave
                # the deployment advertised as `ready`. mark_deployment_undeployed flips the
                # deployment regardless of (and without disturbing) the current state.
                mark_deployment_undeployed(run_id)
        except Exception:
            # Best-effort serving teardown: a failure here must not block the cancel
            # below (the run still flips to cancelled and the training endpoint is GC'd).
            pass
    # Durable path first: stop the exact remote worker via the handle's provider
    # (works from any process); endpoint/instance teardown is shared with the GC.
    # Dispatched generically through the registry — never a hardcoded per-provider branch.
    if remote:
        try:
            from flash.providers import get_provider
            from flash.providers.base import JobHandle

            handle = JobHandle.from_dict(remote)
            provider = get_provider(handle.provider)
            provider.cancel(handle)
            # Vast bills until destroyed, so also belt-and-suspenders destroy the
            # instance (a no-op cost-wise for runpod, whose endpoint GC follows).
            provider.destroy(handle)
        except Exception:
            # Best-effort remote stop; _gc_run_endpoints below still tears the endpoint down.
            pass
    _gc_run_endpoints(spec)
    # Final transition to `cancelled`. The run was NON-terminal at entry, but teardown takes
    # time and a terminal state can race in mid-teardown. We must distinguish two cases:
    #
    #   - A concurrent mark_undeployed() (an external `DELETE /v1/runs/{id}/deploy`) flipped a
    #     `deployed` run to terminal `done`. That `done` is NOT a fresh result — it just
    #     restored the run's pre-deploy completion marker while retiring serving. The user
    #     explicitly asked to cancel, so this must be OVERRIDDEN to `cancelled`.
    #   - A genuine training-COMPLETION `done` from the run's own training thread
    #     (_run_job_inner / attach_run), which persisted real metrics+cost+artifacts. Cancel
    #     must NEVER clobber that — the run finished, so the real result is preserved.
    #
    # These two races are mutually exclusive on the entry state: only a `deployed` run owns a
    # deployment that mark_undeployed can race, and only a non-deployed (running/provisioning/
    # queued) run has an in-flight training thread that can complete mid-teardown. So scope the
    # terminal override to runs that were `deployed` at entry — there a racing `done` is always
    # an undeploy artifact (cancel wins); elsewhere a racing `done` is a genuine completion that
    # _update's CAS correctly protects (cancel loses to a real finish).
    _update(run_id, "cancelled", allow_from_terminal=entered_deployed)
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

    spec = JobSpec.from_dict(status.spec)
    remote = dict(status.remote)
    seed = int(remote.pop("seed", spec.train.seeds[0]))
    # The class the run actually provisioned (a policy retry may have walked past the
    # provisional spec.gpu.type). The in-process success path stamps this into metrics;
    # on recovery the worker output carries no such field, so recover it from the handle
    # to cost the right card.
    allocated_gpu = remote.pop("allocated_gpu", None)
    log = log_stream or sys.stderr
    # Dispatch the poll generically via the handle's provider (the provider owns its
    # heartbeat reader + poll loop); the orchestrator stays provider-agnostic.
    from flash.providers import get_provider
    from flash.providers.base import JobHandle

    handle = JobHandle.from_dict(remote)
    print(f"attaching to {run_id}: provider={handle.provider} {handle.data}", file=log)
    res = get_provider(handle.provider).poll(handle, spec, seed, log=log)
    try:
        # A best-effort cancel deletes the job/instance, which the poller reports as a
        # failure (or a late worker may still succeed) — either way, re-read the state
        # first so a recovery thread can't overwrite the user's terminal `cancelled`.
        if get_status(run_id).state == "cancelled":
            return get_status(run_id)
        if not res.ok and res.failure == "no_capacity":
            # The recovered job is stuck IN_QUEUE on a capacity-starved / throttled class with
            # no usable worker (a control-plane restart landed mid-queue). The fresh-submit path
            # WALKS to the next-cheapest available class on no_capacity; recovery must do the same
            # instead of terminally failing on a transient market condition. Tear down the stale
            # endpoint (its worker never started, so nothing is lost) and resume THIS seed through
            # the supervised submit, which owns the capacity walk + crash budget.
            with contextlib.suppress(Exception):
                from flash.providers import get_provider as _gp

                _gp(handle.provider).destroy(handle)
            try:
                resume_index = list(spec.train.seeds).index(seed)
            except ValueError:
                resume_index = 0
            # Drop the now-dead handle so a cancel / restart in the re-provision gap can't reattach
            # to the deleted endpoint; _run_seed_loop's next on_handle records the fresh one.
            _update(run_id, "running", remote=None)
            print(
                f"recovery: {run_id} seed {seed} was IN_QUEUE with no capacity; "
                "re-provisioning via the capacity walk instead of failing",
                file=log,
                flush=True,
            )
            _run_seed_loop(
                spec, log, start_index=resume_index, prior_cost=float(status.cost_usd or 0.0)
            )
            return get_status(run_id)
        if not res.ok:
            _update(run_id, "failed", error=f"{res.failure}: {res.detail}")
            return get_status(run_id)
        # Carry the provisioned class into metrics so _persist_metrics costs the card the
        # run actually used (the in-process path stamps this; recovery must restore it).
        if allocated_gpu and isinstance(res.metrics, dict):
            res.metrics.setdefault("allocated_gpu", allocated_gpu)
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
        # Mirror _run_job: GC any endpoint a transient destroy left behind rather than
        # leaking a billable RunPod endpoint.
        _gc_run_endpoints(spec)
    return get_status(run_id)


def mark_deployed(run_id: str, deployment: dict, expect_state: str | None = None) -> RunStatus:
    # Atomic + terminal-respecting (same guard as _update): a /cancel landing during
    # always-on provisioning/warmup writes `cancelled`; this must NOT overwrite it with
    # `deployed` and resurrect the run as an active deployment. `done` is deployable
    # though (the common case: deploy a finished run), so only the non-`done` terminal
    # states block here — otherwise a freshly finished run could never be deployed.
    #
    # expect_state is a compare-and-set: the deploy flow passes the state it expects the
    # run to still be in (the pre-deploy snapshot, or "deployed" after the provisional
    # mark). If an undeploy raced finalization — deleting the endpoint and writing `done`
    # with deployment.state="undeployed" mid-warmup — the state no longer matches and we
    # refuse to re-advertise the just-deleted endpoint.
    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.state in _UNDEPLOYABLE_STATES:
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        status.deployment = deployment
        status.state = "deployed"
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_undeployed(run_id: str) -> RunStatus:
    """Record an explicit undeploy (endpoint torn down -> run back to `done`).

    Lock-guarded so it serializes with a racing deploy finalization: the raw read +
    _save_status the endpoint used to do could interleave with mark_deployed and be
    clobbered. With this under the same lock, mark_deployed's expect_state CAS then sees
    the `done`/undeployed write and won't re-advertise the deleted endpoint.
    """
    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment:
            status.deployment = {**status.deployment, "state": "undeployed"}
        # Record the teardown but don't resurrect a terminal run: undeploying a
        # cancelled/failed run keeps its terminal state (only a live `deployed` run goes
        # back to `done`). `done` is terminal too, so this naturally no-ops the state.
        if status.state not in TERMINAL_STATES:
            status.state = "done"
        status.updated_at = time.time()
        _save_status(status)
        return status


def mark_deployment_undeployed(run_id: str) -> RunStatus:
    """Flip ONLY the deployment field to ``undeployed``, leaving the run's state untouched.

    Used by ``cancel_run`` to retire a deployed run's serving record. Unlike
    ``mark_undeployed`` (which is a state transition: a live `deployed` run goes back to
    `done`), this never asserts or changes the run state. That matters under the cancel
    race: a concurrent ``mark_undeployed`` may have already moved the run to terminal
    `done`, and ``_update``'s compare-and-set rejects any transition off a terminal state —
    even re-asserting `deployed` to carry the deployment field — which would leave the
    deployment advertised as `ready`. Marking the field directly (lock-guarded for
    serialization) sidesteps the CAS so the deployment reliably ends `undeployed`, while the
    trailing ``cancelled`` transition is left to ``_update``.
    """
    with _STATUS_LOCK:
        status = get_status(run_id)
        if status.deployment:
            status.deployment = {**status.deployment, "state": "undeployed"}
            status.updated_at = time.time()
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
    from flash.providers.runpod.train import upload_code

    # A cancel can land between the queued status being returned to the client and
    # this background thread starting; don't overwrite a terminal state (cancelled)
    # with provisioning and then launch a paid seed as if the cancel never happened.
    if get_status(spec.run_id).state in TERMINAL_STATES:
        return
    _update(spec.run_id, "provisioning")
    log_path = os.path.join(RUNS_DIR, f"{spec.run_id}.log")
    try:
        _run_job_inner(spec, log_path, upload_code)
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
    """Run one seed with the job submit/poll path + bounded auto-retry.

    Each attempt first ALLOCATES the GPU: the cheapest class across providers (RunPod
    live pricing + Vast verified-datacenter offers) that fits the model — re-resolved
    fresh per attempt because offers are a live market. A policy ``gpu.requested``
    ("cheapest"/"auto") lets the allocator pick the class; a concrete ``gpu.requested``
    pins the class (the allocator then only picks the provider); ``gpu.provider`` pins
    the substrate.

    Retries (fresh job on a fresh host; worker resumes from the latest HF
    checkpoint) when the failure looks infra-shaped: a stall (heartbeat frozen), a
    client polling breakdown, or a platform TIMED_OUT/worker-loss. Sick Vast machines
    are blacklisted for the run; failover naturally crosses providers.
    Genuine worker errors (the run's code crashed; traceback persisted to HF) fail
    immediately. The offline test/CI marker FLASH_SKIP_NET takes the blocking
    in-process submit instead (the job poll path is network-only).
    """
    from flash.providers.base import PollResult
    from flash.providers.runpod.train import submit_train

    if os.environ.get("FLASH_SKIP_NET"):
        return submit_train(spec, seed, log=log)

    from flash.providers import get_provider
    from flash.providers.allocator import allocate, allocation_summary
    from flash.providers.base import POLICY_NAMES

    last_handle: dict = {}
    # The friendly GPU class the CURRENT attempt provisioned (set right before each submit),
    # so on_handle persists it into the run handle and a recovery via attach_run costs the
    # class actually used rather than the parse-time provisional spec.gpu.type.
    current_gpu: dict = {}
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
        _update(
            spec.run_id,
            "running",
            remote={**handle, "seed": int(seed), "allocated_gpu": current_gpu.get("name")},
        )

    def _gc_seen_endpoints() -> None:
        if not seen_endpoints:
            return
        from flash.providers.runpod import api as runpod_api

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
    # Index into the ranked candidate list. It advances only after an attempt that
    # actually provisioned a class lost it to an infra failure (see the retry tail), so a
    # failed allocation — which never tried a card — can't skip past the cheapest class.
    gpu_walk_offset = 0
    # Two independent budgets. ``crash_retries`` (bounded by max_retries) covers genuine
    # infra flakes (host died, network timeout) where we re-provision the SAME-or-next class.
    # ``capacity walks`` are different: a ``no_capacity`` (throttled / stuck IN_QUEUE) result
    # just means "this class has no free workers right now" — so we WALK to the next-cheapest
    # candidate WITHOUT spending the crash budget and WITHOUT blacklisting the class (it may
    # free up; we simply prefer one that's available now). Bounded by the candidate count so a
    # fully-throttled market terminates instead of looping forever.
    crash_retries = 0
    attempt = -1
    while True:
        attempt += 1
        if attempt > 0 and last_handle:
            # A stalled/timed-out attempt often means the worker is pinned to a
            # throttled/sick host; tear it down so the fresh deploy lands elsewhere.
            # Dispatched generically via the handle's provider.
            if last_handle.get("provider") == "vast":
                with contextlib.suppress(Exception):
                    from flash.providers import get_provider
                    from flash.providers.base import JobHandle

                    get_provider("vast").destroy(JobHandle.from_dict(last_handle))
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
                    from flash.providers.runpod import api as runpod_api

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
        chosen = None
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
                # Multi-GPU: the allocator must search for offers with this many GPUs (Vast) — else
                # it builds the offer book at num_gpus=1 and a disaggregated run rents a 1-GPU
                # instance (the real cause of "container has 1 GPU"). RunPod gets gpu_count on the
                # endpoint separately (deploy_train_endpoint).
                gpu_count=int(getattr(spec.gpu, "count", 1)),
                # Pass the run's train knobs + thinking so the VRAM estimate reflects THIS job's
                # max_length / group_size / batch_size / lora_rank (and the seq escalation) instead
                # of the generic defaults — else a long-context / big-group run is sized at seq=1024
                # and OOMs the card it picks.
                train=spec.train,
                thinking=spec.thinking,
            )
        except Exception as exc:
            from flash.providers.base import UnsupportedGpuError

            if isinstance(exc, UnsupportedGpuError):
                raise  # config-shaped: no GPU anywhere can run this job
            res = PollResult(False, failure="poll_error", detail=f"allocation: {exc}")
        if alloc is not None:
            # allocate() above ran a live-market price walk; re-check cancellation
            # right before provisioning so a cancel during allocation doesn't still
            # launch a paid worker.
            with contextlib.suppress(FileNotFoundError):
                if get_status(spec.run_id).state == "cancelled":
                    raise _RunCancelled(f"run {spec.run_id} was cancelled")
            # Walk down the ranked candidates by the walk offset (clamped to the last): the
            # first attempt takes the cheapest; each retry that provisioned a class and lost
            # it to an infra failure steps to the next-cheapest, so a capacity-starved class
            # can't burn the whole budget. A concrete pin yields a single candidate, so the
            # clamp keeps a pinned run on its class.
            chosen = alloc.candidates[min(gpu_walk_offset, len(alloc.candidates) - 1)]
            print(allocation_summary(alloc), file=log, flush=True)
            if chosen.gpu != alloc.gpu:
                print(
                    f"retry {attempt}: walking past the cheapest class to {chosen.gpu} "
                    f"@ ${chosen.hourly_usd:.2f}/hr",
                    file=log,
                    flush=True,
                )
            run_spec = _spec_with_gpu(spec, chosen.gpu)
            current_gpu["name"] = chosen.gpu
            provider = get_provider(chosen.provider)
            # Vast needs the live-market offer book for the chosen class first, then the
            # other allocator-approved classes by price; RunPod ignores ``offers``.
            offers = None
            if chosen.provider == "vast":
                ok_classes = {c.gpu for c in alloc.candidates if c.provider == "vast"}
                offers = sorted(
                    (o for o in alloc.provider_offers if o.gpu in ok_classes),
                    key=lambda o: (o.gpu != chosen.gpu, o.dph_total),
                )
            try:
                res = provider.submit_run(
                    run_spec,
                    seed,
                    log=log,
                    on_handle=on_handle,
                    attempt=attempt,
                    offers=offers,
                    # The run's machine blacklist must reach the provider so an in-provider
                    # offer REFRESH (Vast) keeps stalled/sick machines excluded.
                    exclude_machine_ids=frozenset(bad_machines),
                )
            except Exception as exc:
                # Deploy/submit themselves can fail transiently (observed: RunPod
                # GraphQL "Something went wrong" x3 during a retry deploy; a vast offer
                # pool emptying between search and rent). That must consume a retry, not
                # kill the run — the budget exists precisely for flakes.
                res = PollResult(False, failure="poll_error", detail=f"deploy/submit: {exc}")
                if crash_retries < max_retries:
                    time.sleep(10 * (crash_retries + 1))  # let the transient clear
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
            if chosen is not None and isinstance(res.metrics, dict):
                res.metrics.setdefault("allocated_gpu", chosen.gpu)
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
            # Host vanished mid-run: the instance went "missing"/dead and NOTHING was captured
            # (no marker error, no error_<phase>.txt, no console log) so _failure_detail falls back
            # to this bare sentinel. A genuine worker code crash instead yields a RICHER detail
            # (the captured traceback), so this exact phrase only ever marks a dead host -> retry it
            # on a fresh one. Without this, a single ~1-in-200 host death killed the whole run.
            "terminated without a DONE sentinel",
        )
        infra_shaped = res.failure in ("stalled", "poll_error", "no_capacity") or any(
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
        # A capacity walk (no_capacity: the class is throttled / has no free workers right now)
        # hops to the next-cheapest candidate WITHOUT spending the crash-retry budget and WITHOUT
        # blacklisting the class — we just prefer one that's available now. It stops only once
        # every candidate class has been tried (a fully-throttled market). A genuine infra crash
        # (host died / network) instead spends the bounded crash budget.
        is_capacity = res.failure == "no_capacity"
        ncand = len(alloc.candidates) if alloc is not None else 0
        if is_capacity:
            can_retry = infra_shaped and chosen is not None and gpu_walk_offset < ncand - 1
        else:
            can_retry = infra_shaped and crash_retries < max_retries
        print(
            f"seed={seed} attempt={attempt} failed ({res.failure}); "
            f"{'retrying (resume from last checkpoint)' if can_retry else 'not retrying'}"
            f"\n--- failure detail ---\n{(res.detail or '')[:2000]}\n---",
            file=log,
            flush=True,
        )
        if not can_retry:
            break
        # Advance to the next-cheapest candidate when THIS attempt actually provisioned one.
        # An allocation/pricing failure (chosen is None) never tried a card, so the next attempt
        # must retry from the cheapest, not walk past it.
        if chosen is not None:
            gpu_walk_offset += 1
            if is_capacity:
                print(
                    f"retry: {chosen.gpu} had no free workers (throttled / capacity-starved); "
                    "walking to the next-cheapest available class (not blacklisting it)",
                    file=log,
                    flush=True,
                )
        if not is_capacity:
            crash_retries += 1
    # Retry budget exhausted: GC every endpoint this seed registered (the final
    # attempt's is in status.remote for _gc_run_endpoints, but intermediate rN ones
    # are only known here).
    _gc_seen_endpoints()
    raise RuntimeError(f"seed {seed} failed after retries: {last_detail}")


def _run_job_inner(spec: JobSpec, log_path: str, upload_code) -> None:
    try:
        # Ship the slm package to the run's HF repo (the per-run [train] hf_repo) so the GPU
        # worker — which fetches code/** from that same repo — can run it.
        upload_code(spec.train.hf_repo)
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
            from flash.providers import get_provider
            from flash.providers.base import JobHandle

            handle = JobHandle.from_dict(status.remote)
            get_provider(handle.provider).destroy(handle)
        except Exception:
            # Best-effort GC; the name-reconstructed RunPod gc below is the backstop.
            pass
    try:
        # RunPod's gc reaps rN-suffixed endpoints the persisted handle can't name.
        from flash.providers import get_provider

        get_provider("runpod").gc(spec)
    except Exception:
        # Best-effort GC; an undeleted endpoint only holds worker quota, never blocks the run.
        pass
    # Vast instances bill until destroyed: the runner's per-attempt `finally` already
    # destroys them, but a crashed supervisor thread can leave one behind. Reap any
    # instance still labeled for this run via the provider's gc (best-effort).
    from flash.providers import available_providers, get_provider

    if "vast" in available_providers():
        with contextlib.suppress(Exception):
            get_provider("vast").gc(spec)


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
    # _gpu_rate is the PER-CARD RunPod price; a disaggregated run rents [gpu] count cards
    # (trainer + inference), so the wall-based projection below must rate the whole node, not
    # one card. (Vast stamps its own total cost_usd from the offer and short-circuits this.)
    rate = _gpu_rate(gpu_type) * max(1, int(getattr(spec.gpu, "count", 1)))
    # A non-runpod provider (e.g. Vast) stamps the real cost_usd from its offer's $/hr
    # AND tags notes["provider"] with its own name — and a near-zero-duration run can
    # legitimately stamp cost_usd == 0.0. The RunPod arm, by contrast, never stamps a real
    # cost: it arrives with cost_usd absent (or a 0.0 placeholder) and no provider note, so
    # the wall-based projection below must run. A bare `cost or 0.0` would treat the Vast
    # 0.0 as "absent" and re-rate it against RunPod pricing while overwriting the provider
    # notes, mis-attributing the run to 'runpod'. So fall back only when the cost is
    # missing/zero AND it has NOT already been attributed to a non-runpod provider.
    _notes = metrics.get("notes")
    _stamped_provider = _notes.get("provider") if isinstance(_notes, dict) else None
    _non_runpod = bool(_stamped_provider) and _stamped_provider != "runpod"
    cost = metrics.get("cost_usd")
    if cost or _non_runpod:
        cost = float(cost or 0.0)
    else:
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


def _update(run_id: str, state: str, *, allow_from_terminal: bool = False, **updates) -> None:
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
        #
        # allow_from_terminal is the NARROW escape hatch used ONLY by cancel_run's final
        # `cancelled` transition, and ONLY when the run was `deployed` at cancel entry (see
        # cancel_run). In that case an explicit user cancel must WIN over a racing
        # mark_undeployed() that flipped the `deployed` run to terminal `done` mid-teardown —
        # that `done` is an undeploy artifact (restoring the pre-deploy completion marker while
        # retiring serving), not a fresh result. Without the override the `cancelled` write
        # no-ops against the freshly-written `done` and the run wrongly ends `done` despite the
        # user asking to cancel. cancel_run passes allow_from_terminal=False for a non-deployed
        # run, so a GENUINE training-completion `done` racing in from the run's own training
        # thread is protected by the CAS below — cancel correctly loses to a real finish.
        if status.state in TERMINAL_STATES and state != status.state and not allow_from_terminal:
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
