"""Durable run primitives: explicit deploy -> submit -> poll with a persisted job handle.

The legacy path (`runpod_flash`'s all-in-one blocking handler call) ties a run's life to
one client process and one HTTP poll loop: a client crash/network blip orphans an
otherwise-healthy GPU job (no job id is ever persisted), and any SDK polling bug kills
the run. This module owns the lifecycle instead:

  deploy_train_endpoint()  -> endpoint_id (Flash SDK deploy, same worker template)
  build_function_input()   -> the exact FunctionRequest payload Flash workers expect
  submit + poll_job()      -> REST queue API with hardened retries; the job handle
                              {endpoint_id, job_id} is persisted by the orchestrator so
                              any process can re-attach (`slm attach`).

Worker compatibility is preserved exactly: the same `_train_body` source is shipped,
the same dependencies are installed, and the same metrics dict is returned.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from dataclasses import dataclass

from autoslm._logging import get_logger
from autoslm.flash import runpod_api
from autoslm.flash.gpus import canonical_gpu, flash_gpu
from autoslm.flash.train import (
    DEFAULT_EXECUTION_TIMEOUT_MS,
    WORKER_SYSTEM_DEPS,
    _patch_runpod_backoff,
    _train_body,
    endpoint_name,
    isolate_flash_state,
    min_cuda_for,
    resolve_worker_deps,
)

logger = get_logger(__name__)

TERMINAL_OK = {"COMPLETED"}
TERMINAL_FAIL = {"FAILED", "CANCELLED", "TIMED_OUT"}


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
    checkpoint alone exceeds 64 GB (e.g. Qwen3.6-35B-A3B at ~72 GB bf16). The template
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
        # `provider` is routing metadata consumed upstream (orchestrator); handles
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

    from autoslm.flash.auth import ensure_auth

    ensure_auth()
    _patch_runpod_backoff()
    isolate_flash_state(name_suffix)
    friendly = canonical_gpu(friendly_gpu)
    name = endpoint_name(friendly, name_suffix)
    kwargs = dict(
        name=name,
        gpu=flash_gpu(friendly),
        gpu_count=1,
        min_cuda_version=min_cuda_for(friendly),
        execution_timeout_ms=execution_timeout_ms or DEFAULT_EXECUTION_TIMEOUT_MS,
        workers=(0, 1),
        **volume_endpoint_kwargs(spec),
    )
    image = os.environ.get("AUTOSLM_WORKER_IMAGE")
    if image:
        kwargs["image"] = image
    else:
        kwargs["dependencies"] = resolve_worker_deps()
        kwargs["system_dependencies"] = WORKER_SYSTEM_DEPS
    ep = Endpoint(**kwargs)
    ep._qb_target = _train_body
    config = ep._build_resource_config()
    apply_disk_gb(config, disk_gb)
    from runpod_flash.core.resources.resource_manager import ResourceManager

    rm = ResourceManager()
    resource = asyncio.run(rm.get_or_deploy_resource(config))
    endpoint_id = getattr(resource, "id", None)
    if not endpoint_id:
        raise RuntimeError(f"deploy_train_endpoint: no endpoint id on resource {resource!r}")
    return endpoint_id, name


def build_function_input(payload: dict) -> dict:
    """The FunctionRequest dict a Flash queue worker expects for `_train_body(payload)`."""
    from runpod_flash.runtime.serialization import serialize_args
    from runpod_flash.stubs.live_serverless import get_function_source

    source, _src_hash = get_function_source(_train_body)
    req: dict = {
        "function_name": "_train_body",
        "function_code": source,
        "args": serialize_args((payload,)),
        "accelerate_downloads": True,
    }
    if not os.environ.get("AUTOSLM_WORKER_IMAGE"):
        req["dependencies"] = resolve_worker_deps()
        req["system_dependencies"] = WORKER_SYSTEM_DEPS
    return req


def submit(endpoint_id: str, payload: dict) -> str:
    return runpod_api.submit_job(endpoint_id, build_function_input(payload))


def decode_output(output) -> dict:
    """Decode a Flash FunctionResponse job output into the worker's metrics dict."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"unexpected job output: {output[:200]}") from exc
    if not isinstance(output, dict):
        raise RuntimeError(f"unexpected job output type: {type(output)}")
    if output.get("success") and output.get("result") is not None:
        import cloudpickle

        result = cloudpickle.loads(base64.b64decode(output["result"]))
        if not isinstance(result, dict):
            raise RuntimeError(f"flash job returned no metrics: {result!r}")
        return result
    err = output.get("error") or "unknown worker error"
    stdout_tail = (output.get("stdout") or "")[-1500:]
    raise RuntimeError(f"Remote execution failed: {err}\n--- worker stdout tail ---\n{stdout_tail}")


@dataclass
class PollResult:
    ok: bool
    metrics: dict | None = None
    failure: str | None = None  # "job_failed" | "stalled" | "poll_error"
    detail: str | None = None


def poll_job(
    handle: JobHandle,
    log=None,
    interval_s: float = 10.0,
    heartbeat_reader=None,
    stall_after_s: float = 1200.0,
    deadline_s: float | None = None,
) -> PollResult:
    """Poll a queue job to completion; resilient to transient API errors.

    ``heartbeat_reader`` (optional callable -> dict|None) surfaces worker progress into
    the run log and powers stall detection: if the job claims IN_PROGRESS but the
    worker heartbeat hasn't advanced for ``stall_after_s``, we declare a stall so the
    supervisor can resubmit instead of waiting for the wall-clock cap.
    """

    def say(msg: str):
        if log is not None:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log, flush=True)

    start = time.time()
    last_status = None
    last_hb_key = None
    last_progress = time.time()
    last_health_probe = 0.0
    consecutive_poll_errors = 0
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
            return PollResult(False, failure="stalled", detail="client-side deadline exceeded")
        try:
            st = runpod_api.job_status(handle.endpoint_id, handle.job_id)
            consecutive_poll_errors = 0
        except runpod_api.RunpodApiError as e:
            consecutive_poll_errors += 1
            say(f"poll error ({consecutive_poll_errors}): {e}")
            if consecutive_poll_errors >= 8:
                return PollResult(False, failure="poll_error", detail=str(e))
            time.sleep(min(60, interval_s * consecutive_poll_errors))
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
            # Prefix the terminal status so the orchestrator's infra-retry markers
            # (e.g. TIMED_OUT) match even when RunPod sets no error/output text.
            return PollResult(False, failure="job_failed", detail=f"[{status}] {detail}")
        # While queued, surface worker availability (throttled hosts are the common
        # cause of silent multi-minute waits — make them visible in the run log).
        if status == "IN_QUEUE" and time.time() - last_health_probe > 90:
            last_health_probe = time.time()
            try:
                h = runpod_api.endpoint_health(handle.endpoint_id)
                workers = h.get("workers") or {}
                if any(workers.get(k) for k in ("throttled", "unhealthy", "initializing")) or not (
                    workers.get("running") or workers.get("ready") or workers.get("idle")
                ):
                    say(f"queued; workers: {workers}")
            except Exception:
                # Health surfacing is diagnostic only; a probe failure must not stop polling.
                pass
        # heartbeat progress surfacing + stall detection
        if heartbeat_reader is not None:
            try:
                hb = heartbeat_reader()
            except Exception:
                hb = None
            if hb:
                key = (hb.get("stage"), hb.get("step"), hb.get("ts"))
                if key != last_hb_key:
                    last_hb_key = key
                    last_progress = time.time()
                    stage = hb.get("stage")
                    step = hb.get("step")
                    reward = hb.get("reward")
                    say(
                        f"worker: stage={stage}"
                        + (f" step={step}" if step is not None else "")
                        + (f" reward={reward:.3f}" if isinstance(reward, int | float) else "")
                    )
        if time.time() - last_progress > stall_after_s:
            return PollResult(
                False,
                failure="stalled",
                detail=f"no worker progress for {int(time.time() - last_progress)}s "
                f"(job status {status})",
            )
        time.sleep(interval_s)


def submit_train_durable(spec, seed: int, log=None, on_handle=None, attempt: int = 0) -> PollResult:
    """Durable equivalent of ``submit_train``: deploy, submit, persist handle, poll.

    ``on_handle(handle_dict)`` is invoked as soon as the job is queued so the
    orchestrator can persist {endpoint_id, job_id} for cross-process reattach.
    """
    from autoslm.envs.registry import worker_pip_for_env
    from autoslm.flash.train import _run_suffix, build_worker_env

    timeout_s = max(60, int(spec.gpu.max_wall_seconds))
    # Per-attempt endpoint name: a retry must land on a genuinely fresh endpoint —
    # reusing the name lets the SDK/platform pin the job back onto the same
    # (possibly throttled/sick) host.
    suffix = _run_suffix(spec.run_id)
    if attempt:
        suffix = f"{suffix}r{attempt}"
    endpoint_id, name = deploy_train_endpoint(
        spec.gpu.type,
        execution_timeout_ms=timeout_s * 1000,
        name_suffix=suffix,
        disk_gb=spec.gpu.disk_gb,
        spec=spec,
    )
    payload = {
        "hf_repo": os.environ.get("HF_REPO", ""),
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "env": build_worker_env(spec, seed),
        "extra_pip": list(spec.environment.pip)
        or worker_pip_for_env(spec.environment.id, spec.environment.params),
    }
    try:
        job_id = submit(endpoint_id, payload)
    except Exception:
        # The endpoint is registered but no durable handle exists yet, and a
        # retry endpoint's rN-suffixed name can't be reconstructed from the run
        # id later — delete it now so a transient submit failure doesn't leak a
        # serverless endpoint against the account quota.
        with contextlib.suppress(Exception):
            runpod_api.delete_endpoint(endpoint_id)
        raise
    handle = JobHandle(endpoint_id, name, job_id)
    if log is not None:
        print(
            f"submitted durable job: endpoint={name} ({endpoint_id}) job={job_id} "
            f"attempt={attempt} gpu={spec.gpu.type} phase={spec.phase} seed={seed}",
            file=log,
            flush=True,
        )
    if on_handle is not None:
        on_handle(handle.to_dict())
    hf_repo = os.environ.get("HF_REPO", "")
    prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
    reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
    stall = float(os.environ.get("AUTOSLM_STALL_AFTER_S", "1500"))
    return poll_job(handle, log=log, heartbeat_reader=reader, stall_after_s=stall)


def make_hf_heartbeat_reader(hf_repo: str, prefix: str, min_interval_s: float = 30.0):
    """Reader for the worker's heartbeat.json on HF (rate-limited, never raises)."""
    state = {"last": 0.0}

    def read() -> dict | None:
        if time.time() - state["last"] < min_interval_s:
            return None
        state["last"] = time.time()
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(
                hf_repo,
                f"{prefix}/heartbeat.json",
                repo_type="dataset",
                token=os.environ.get("HUGGINGFACE_TOKEN"),
                force_download=True,
            )
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    return read
