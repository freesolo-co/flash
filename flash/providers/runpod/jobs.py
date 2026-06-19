"""Durable run primitives: explicit deploy -> submit -> poll with a persisted job handle.

Calling `runpod_flash`'s all-in-one blocking handler directly would tie a run's life to
one client process and one HTTP poll loop: a client crash/network blip orphans an
otherwise-healthy GPU job (no job id is ever persisted), and any SDK polling bug kills
the run. This module owns the lifecycle instead:

  deploy_train_endpoint()  -> endpoint_id (Flash SDK deploy, same worker template)
  build_function_input()   -> the exact FunctionRequest payload Flash workers expect
  submit + poll_job()      -> REST queue API with hardened retries; the job handle
                              {endpoint_id, job_id} is persisted by the runner so
                              any process can re-attach (`slm attach`).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from dataclasses import dataclass

from flash._logging import get_logger
from flash.providers._poll import PollErrorTracker, make_say, surface_heartbeat
from flash.providers.base import PollResult, canonical_gpu
from flash.providers.runpod import api as runpod_api
from flash.providers.runpod.gpus import flash_gpu
from flash.providers.runpod.train import (
    DEFAULT_EXECUTION_TIMEOUT_MS,
    FLASH_SDK_LOCK,
    WORKER_IMAGE,
    WORKER_SYSTEM_DEPS,
    _patch_runpod_backoff,
    _train_body,
    endpoint_name,
    isolate_flash_state,
    min_cuda_for,
    resolve_worker_deps,
)

logger = get_logger(__name__)

# Re-export so callers/tests that did ``from ...jobs import PollResult`` keep working.
__all__ = [
    "JobHandle",
    "PollResult",
    "apply_disk_gb",
    "build_function_input",
    "decode_output",
    "deploy_train_endpoint",
    "make_hf_heartbeat_reader",
    "make_hf_text_reader",
    "poll_job",
    "submit_run",
    "volume_endpoint_kwargs",
]

TERMINAL_OK = {"COMPLETED"}
TERMINAL_FAIL = {"FAILED", "CANCELLED", "TIMED_OUT"}

# Heartbeat stages the worker emits DURING cold start, BEFORE the model is loaded and the
# training loop begins (boot -> sft_start/rl_start, then later sft_model_load/rl_train_start).
# Receiving one proves the worker is alive but NOT that the slow setup (model download +
# vLLM init) finished, so they must not flip stall detection to the tight training window.
_SETUP_HEARTBEAT_STAGES = frozenset(
    {"boot", "sft_start", "rl_start", "sft_model_load", "rl_train_start"}
)


def stall_kwargs() -> dict:
    """``poll_job`` stall-window kwargs, shared by the submit and reattach paths so a recovered
    run uses the same tuning as the original submit. ``stall_after_s`` = post-training-heartbeat
    window; ``setup_grace_s`` = the larger cold-start window before the first training heartbeat.
    """
    return {"stall_after_s": 1500.0, "setup_grace_s": 3000.0}


def volume_endpoint_kwargs(spec) -> dict:
    """Endpoint kwargs for the OPT-IN persistent network volume (cross-run HF cache).

    Returns {} unless ``gpu.network_volume`` is set. The volume pins the endpoint to
    one datacenter (``gpu.datacenter``, default EU-RO-1 — the SDK's storage default),
    which shrinks the available GPU pool; that trade-off is why this is opt-in.
    """
    nv = getattr(spec.gpu, "network_volume", None) if spec is not None else None
    if not nv:
        return {}
    from runpod_flash import NetworkVolume
    from runpod_flash.core.resources.datacenter import DataCenter

    dc = DataCenter.from_string(spec.gpu.datacenter) if spec.gpu.datacenter else None
    volume = NetworkVolume(
        name=str(nv),
        size=int(getattr(spec.gpu, "network_volume_gb", 100) or 100),
        **({"datacenter": dc} if dc else {}),
    )
    kwargs: dict = {"volume": volume}
    if dc:
        kwargs["datacenter"] = dc
    return kwargs


def apply_disk_gb(config, disk_gb: int | None) -> None:
    """Raise the worker's container disk on a built endpoint config.

    The Flash SDK's ``PodTemplate.containerDiskInGb`` defaults to 64 GB and the
    ``Endpoint`` wrapper exposes no disk knob, which is what blocked models whose
    checkpoint alone exceeds 64 GB. The template
    is already populated by the SDK's validators when the resource config is built, so
    raising the field here is the supported injection point. Raise-only: shrinking
    below the SDK default buys nothing (serverless disk isn't billed separately) and
    would regress runs whose configs carry the historical ``disk_gb = 60`` default.
    """
    if not disk_gb:
        return
    template = getattr(config, "template", None)
    if template is None:
        logger.warning("disk_gb=%s requested but endpoint config has no template", disk_gb)
        return
    template.containerDiskInGb = max(int(disk_gb), int(template.containerDiskInGb or 0))


@dataclass
class JobHandle:
    endpoint_id: str
    endpoint_name: str
    job_id: str

    def to_dict(self) -> dict:
        return {
            "provider": "runpod",
            "endpoint_id": self.endpoint_id,
            "endpoint_name": self.endpoint_name,
            "job_id": self.job_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> JobHandle:
        # `provider` is routing metadata consumed upstream (runner); handles
        # persisted before it existed default to runpod there.
        return cls(d["endpoint_id"], d.get("endpoint_name", ""), d["job_id"])


def deploy_train_endpoint(
    friendly_gpu: str,
    execution_timeout_ms: int | None = None,
    name_suffix: str | None = None,
    disk_gb: int | None = None,
    spec=None,
) -> tuple[str, str]:
    """Deploy (or reuse) the run's uniquely-named worker endpoint; return (id, name)."""
    os.environ["FLASH_IS_LIVE_PROVISIONING"] = "true"
    from runpod_flash import Endpoint

    from flash.providers.runpod.auth import ensure_auth

    ensure_auth()
    _patch_runpod_backoff()
    friendly = canonical_gpu(friendly_gpu)
    name = endpoint_name(friendly, name_suffix)
    # The baked WORKER_IMAGE is now a self-contained RunPod Serverless worker (its CMD runs
    # rp_handler.py, which reads job["input"] and runs the training) — deploy it directly (Flash
    # "client mode"). build_function_input then sends the payload as the job input. FLASH_WORKER_IMAGE
    # overrides the baked image (e.g. a hotfix tag); since WORKER_IMAGE is a non-empty constant the
    # image is always set, so the boot-install/live-function path is only reachable if both are
    # explicitly cleared (not a normal configuration).
    image = os.environ.get("FLASH_WORKER_IMAGE") or WORKER_IMAGE
    from runpod_flash.core.resources.resource_manager import ResourceManager

    # isolate_flash_state mutates runpod_flash's process-wide registry globals for this run's
    # suffix, and ResourceManager + the deploy share the SDK's asyncio singleton. Hold the
    # lock across the whole critical section so a concurrent run can't swap the registry
    # scope or race the event loop mid-deploy.
    with FLASH_SDK_LOCK:
        isolate_flash_state(name_suffix)
        kwargs = dict(
            name=name,
            gpu=flash_gpu(friendly),
            gpu_count=1,
            min_cuda_version=min_cuda_for(friendly),
            execution_timeout_ms=execution_timeout_ms or DEFAULT_EXECUTION_TIMEOUT_MS,
            workers=(0, 1),
            **volume_endpoint_kwargs(spec),
        )
        if image:
            kwargs["image"] = image
        else:
            # Pass the resolved GPU so Hopper (sm90) gets its fla-drop treatment (resolve_worker_deps
            # is GPU-scoped); a bare call would ship the generic deps and run fla's #640-buggy GDN
            # Triton kernel on an H100 instead of the correct pure-PyTorch delta rule.
            kwargs["dependencies"] = resolve_worker_deps(friendly)
            kwargs["system_dependencies"] = WORKER_SYSTEM_DEPS
        ep = Endpoint(**kwargs)
        ep._qb_target = _train_body
        config = ep._build_resource_config()
        apply_disk_gb(config, disk_gb)
        # Worker image is PUBLIC, so no container-registry credential is needed to pull it.
        rm = ResourceManager()
        resource = asyncio.run(rm.get_or_deploy_resource(config))
    endpoint_id = getattr(resource, "id", None)
    if not endpoint_id:
        raise RuntimeError(f"deploy_train_endpoint: no endpoint id on resource {resource!r}")
    return endpoint_id, name


def build_function_input(payload: dict, friendly_gpu: str | None = None) -> dict:
    """The FunctionRequest dict a Flash queue worker expects for `_train_body(payload)`.

    ``friendly_gpu`` is threaded into ``resolve_worker_deps`` so the request-level dependency
    list is GPU-scoped exactly like the endpoint config (deploy_train_endpoint): on Hopper (sm90)
    it must drop ``flash-linear-attention`` so the worker uses the pure-PyTorch delta rule instead
    of fla's #640-buggy GDN Triton kernel. A bare call would reinstall the generic deps and
    reintroduce that sm90 correctness issue even when the endpoint was configured correctly.
    """
    if os.environ.get("FLASH_WORKER_IMAGE") or WORKER_IMAGE:
        # Baked serverless-worker image (client mode): the image's rp_handler reads job["input"]
        # and calls _train_body, so the job input IS the train payload (submit_job wraps it in
        # {"input": ...}). No live-function source, no boot-install deps.
        return payload
    # Boot-install fallback (Flash default image + live function): ship _train_body's source for the
    # generic worker to run, plus the GPU-scoped deps to install on first use (drops fla on Hopper).
    from runpod_flash.runtime.serialization import serialize_args
    from runpod_flash.stubs.live_serverless import get_function_source

    source, _src_hash = get_function_source(_train_body)
    return {
        "function_name": "_train_body",
        "function_code": source,
        "args": serialize_args((payload,)),
        "accelerate_downloads": True,
        "dependencies": resolve_worker_deps(canonical_gpu(friendly_gpu) if friendly_gpu else None),
        "system_dependencies": WORKER_SYSTEM_DEPS,
    }


def decode_output(output) -> dict:
    """Decode a queue-job output into the worker's metrics dict. Handles BOTH job shapes:

    - Flash LIVE-function (boot-install path): a FunctionResponse envelope
      ``{"success": True, "result": <base64 cloudpickle of the dict>}``.
    - Client-mode SERVERLESS handler (baked-image path): our baked rp_handler returns
      ``_train_body(...)``'s metrics dict, which RunPod surfaces as ``job["output"]`` directly —
      no envelope. The metrics dict has no ``success``/``result`` keys, so we return it as-is.
    """
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"unexpected job output: {output[:200]}") from exc
    if not isinstance(output, dict):
        raise RuntimeError(f"unexpected job output type: {type(output)}")
    # Flash live-function envelope (has success/result/error keys).
    if "success" in output or "result" in output:
        if output.get("success") and output.get("result") is not None:
            import cloudpickle

            result = cloudpickle.loads(base64.b64decode(output["result"]))
            if not isinstance(result, dict):
                raise RuntimeError(f"flash job returned no metrics: {result!r}")
            return result
        err = output.get("error") or "unknown worker error"
        stdout_tail = (output.get("stdout") or "")[-1500:]
        raise RuntimeError(
            f"Remote execution failed: {err}\n--- worker stdout tail ---\n{stdout_tail}"
        )
    # Client-mode serverless handler: the metrics dict IS the output (baked rp_handler).
    if output.get("error"):
        # Mirror the Flash path: append the worker stdout tail when present so poll_job's
        # root-cause diagnostics (e.g. a vLLM crash) survive the client-mode failure shape too.
        stdout_tail = (output.get("stdout") or "")[-1500:]
        msg = f"Remote execution failed: {output['error']}"
        if stdout_tail:
            msg += f"\n--- worker stdout tail ---\n{stdout_tail}"
        raise RuntimeError(msg)
    return output


def poll_job(
    handle: JobHandle,
    log=None,
    interval_s: float = 10.0,
    heartbeat_reader=None,
    stall_after_s: float = 1200.0,
    setup_grace_s: float = 3000.0,
    unhealthy_grace_s: float = 240.0,
    deadline_s: float | None = None,
) -> PollResult:
    """Poll a queue job to completion; resilient to transient API errors.

    Two stall windows: the cold-start phase (dep install, per-run env pip, model download,
    vLLM init) is slow and only emits *setup* heartbeats (``_SETUP_HEARTBEAT_STAGES``).
    Until a *training* heartbeat arrives we apply the larger ``setup_grace_s`` budget so a
    slow cold start isn't misread as a stall; after it we use the tight ``stall_after_s``.
    Needs a ``heartbeat_reader`` to tell the phases apart — without one we keep
    ``stall_after_s`` throughout (no regression).
    """

    say = make_say(log)
    poll_errors = PollErrorTracker(say, interval_s)

    start = time.time()
    last_status = None
    last_hb_key = None
    last_progress = time.time()
    seen_heartbeat = False
    last_health_probe = 0.0
    unhealthy_since: float | None = None  # first time the worker was seen stuck UNHEALTHY
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
            return PollResult(False, failure="stalled", detail="client-side deadline exceeded")
        try:
            st = runpod_api.job_status(handle.endpoint_id, handle.job_id)
            poll_errors.reset()
        except runpod_api.RunpodApiError as e:
            if poll_errors.record(e):
                return PollResult(False, failure="poll_error", detail=str(e))
            continue
        status = st.get("status")
        if status != last_status:
            say(f"job {handle.job_id}: {status}")
            last_status = status
            last_progress = time.time()
        if status in TERMINAL_OK:
            try:
                return PollResult(True, metrics=decode_output(st.get("output")))
            except RuntimeError as e:
                return PollResult(False, failure="job_failed", detail=str(e))
        if status in TERMINAL_FAIL:
            detail = str(st.get("error") or "")[:1500]
            out = st.get("output")
            if isinstance(out, dict) and out.get("stdout"):
                # Worker stdout tail is the only place the REAL root cause lives for
                # crashes inside subprocesses (e.g. vLLM EngineCore deaths).
                detail += "\n--- worker stdout tail ---\n" + str(out["stdout"])[-2000:]
            elif not detail:
                detail = str(out)[:1500]
            # Prefix the terminal status so the runner's infra-retry markers
            # (e.g. TIMED_OUT) match even when RunPod sets no error/output text.
            return PollResult(False, failure="job_failed", detail=f"[{status}] {detail}")
        # While queued, surface worker availability (throttled hosts are the common
        # cause of silent multi-minute waits — make them visible in the run log).
        if status == "IN_QUEUE" and time.time() - last_health_probe > 90:
            last_health_probe = time.time()
            try:
                h = runpod_api.endpoint_health(handle.endpoint_id)
                workers = h.get("workers") or {}
                usable = workers.get("running") or workers.get("ready") or workers.get("idle")
                recovering = workers.get("initializing")
                if any(workers.get(k) for k in ("throttled", "unhealthy", "initializing")) or not usable:
                    say(f"queued; workers: {workers}")
                # Fail fast on a worker stuck UNHEALTHY: a dead worker / failed image pull won't
                # self-recover, so don't burn the full setup_grace_s (~50 min) waiting on it — once
                # it has stayed unhealthy with nothing usable or (re)initializing for
                # unhealthy_grace_s, return a (retryable) stall so the runner re-provisions a FRESH
                # endpoint (fresh image pull, likely a different host). Observed: a mutable image
                # tag republished mid-pull corrupts the worker -> unhealthy, and a fresh pull fixes it.
                if workers.get("unhealthy") and not usable and not recovering:
                    if unhealthy_since is None:
                        unhealthy_since = time.time()
                    elif time.time() - unhealthy_since > unhealthy_grace_s:
                        return PollResult(
                            False,
                            failure="stalled",
                            detail=f"worker stuck unhealthy for "
                            f"{int(time.time() - unhealthy_since)}s while IN_QUEUE (likely a failed "
                            f"image pull); retrying on a fresh endpoint",
                        )
                else:
                    unhealthy_since = None  # recovered / usable worker appeared
            except Exception:
                # Health surfacing is diagnostic only; a probe failure must not stop polling.
                pass
        # heartbeat progress surfacing + stall detection
        new_key, stage = surface_heartbeat(heartbeat_reader, last_hb_key, say)
        if new_key != last_hb_key:
            last_hb_key = new_key
            last_progress = time.time()
            # Only a training-phase heartbeat means cold-start setup is done and we
            # can switch to the tight window; setup heartbeats keep the grace budget.
            if stage not in _SETUP_HEARTBEAT_STAGES:
                seen_heartbeat = True
        # Cold start (before any training-phase heartbeat) gets the larger setup_grace_s,
        # but only when a heartbeat_reader lets us tell setup from training; without one we
        # can't, so stay on stall_after_s (no regression).
        in_setup = heartbeat_reader is not None and not seen_heartbeat
        stall_limit = setup_grace_s if in_setup else stall_after_s
        if time.time() - last_progress > stall_limit:
            phase = "setup (pre-training)" if in_setup else "training"
            return PollResult(
                False,
                failure="stalled",
                detail=f"no worker progress for {int(time.time() - last_progress)}s "
                f"during {phase} (job status {status}, limit {int(stall_limit)}s)",
            )
        time.sleep(interval_s)


def submit_run(spec, seed: int, log=None, on_handle=None, attempt: int = 0) -> PollResult:
    """Durable equivalent of ``submit_train``: deploy, submit, persist handle, poll.

    ``on_handle(handle_dict)`` is invoked as soon as the job is queued so the
    runner can persist {endpoint_id, job_id} for cross-process reattach.
    """
    from flash.envs.registry import worker_hub_env_ids, worker_pip_for_env
    from flash.providers.runpod.train import _run_suffix, build_worker_env, chalk_extra_pip

    timeout_s = max(60, int(spec.gpu.max_wall_seconds))
    # Per-attempt endpoint name: a retry must land on a genuinely fresh endpoint —
    # reusing the name lets the SDK/platform pin the job back onto the same
    # (possibly throttled/sick) host.
    suffix = _run_suffix(spec.run_id)
    if attempt:
        suffix = f"{suffix}r{attempt}"
    # Resolve the worker env BEFORE provisioning: an unrecorded Hub env raises here, and
    # doing it after deploy_train_endpoint() would leak the just-created endpoint (its
    # rN-suffixed name can't be reconstructed from the run id later) against the account
    # quota — the runner would also treat the raise as a retryable poll_error.
    # extra_pip runs for EVERY job here (the durable baked-image path skips resolve_worker_deps /
    # FLASH_WORKER_EXTRA_DEPS in build_function_input, but _train_body always pip-installs
    # extra_pip), so the opt-in chalk spec is appended here to reach default runs.
    extra_pip = (
        list(spec.environment.pip) or worker_pip_for_env(spec.environment.id)
    ) + chalk_extra_pip(spec)
    worker_env = build_worker_env(spec, seed)
    endpoint_id, name = deploy_train_endpoint(
        spec.gpu.type,
        execution_timeout_ms=timeout_s * 1000,
        name_suffix=suffix,
        disk_gb=spec.gpu.disk_gb,
        spec=spec,
    )
    payload = {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "env": worker_env,
        "extra_pip": extra_pip,
        "hub_env_ids": worker_hub_env_ids(spec.environment.id, spec.environment.params),
    }
    try:
        job_id = runpod_api.submit_job(endpoint_id, build_function_input(payload, spec.gpu.type))
    except Exception:
        # The endpoint is registered but no run handle exists yet, and a
        # retry endpoint's rN-suffixed name can't be reconstructed from the run
        # id later — delete it now so a transient submit failure doesn't leak a
        # serverless endpoint against the account quota.
        with contextlib.suppress(Exception):
            runpod_api.delete_endpoint(endpoint_id)
        raise
    handle = JobHandle(endpoint_id, name, job_id)
    if log is not None:
        print(
            f"submitted job: endpoint={name} ({endpoint_id}) job={job_id} "
            f"attempt={attempt} gpu={spec.gpu.type} phase={spec.phase} seed={seed}",
            file=log,
            flush=True,
        )
    if on_handle is not None:
        on_handle(handle.to_dict())
    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
    reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
    return poll_job(handle, log=log, heartbeat_reader=reader, **stall_kwargs())


def make_hf_text_reader(hf_repo: str, path_in_repo: str, min_interval_s: float = 45.0):
    """Rate-limited reader for one HF artifact's text content (None until it exists).

    Generic helper shared by both providers' pollers (runpod heartbeats + vast's
    DONE/metrics/error artifacts). ``read(force=False)`` re-downloads at most once per
    ``min_interval_s`` (``force=True`` bypasses the gate); it never raises — any HF error
    (artifact absent, network) returns None.
    """
    state = {"last": 0.0}

    def read(force: bool = False) -> str | None:
        if not hf_repo:
            return None
        if not force and time.time() - state["last"] < min_interval_s:
            return None
        state["last"] = time.time()
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(
                hf_repo,
                path_in_repo,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
                force_download=True,
            )
            with open(p) as f:
                return f.read()
        except Exception:
            return None

    return read


def make_hf_heartbeat_reader(hf_repo: str, prefix: str, min_interval_s: float = 30.0):
    """Reader for the worker's heartbeat.json on HF (rate-limited, never raises).

    Thin JSON-parsing wrapper over :func:`make_hf_text_reader` bound to ``{prefix}/heartbeat.json``.
    """
    text_reader = make_hf_text_reader(hf_repo, f"{prefix}/heartbeat.json", min_interval_s)

    def read() -> dict | None:
        raw = text_reader()
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    return read
