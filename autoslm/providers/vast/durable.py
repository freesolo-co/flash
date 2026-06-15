"""Vast.ai run lifecycle: verified-datacenter offers -> instance -> HF-artifact poll.

The Vast equivalent of ``providers/runpod/durable.py``. Vast has no serverless queue:
we rent a single-GPU instance from a VERIFIED DATACENTER offer, ship a self-contained
bootstrap (the private ``_bootstrap`` module) through the onstart script, and detect
completion purely via the worker's HF artifacts (DONE/metrics.json/heartbeat.json) +
the instance's status — no inbound network to the box is ever needed.

The instance bootstrap is an INTERNAL detail of this module (``build_onstart`` reads
``_bootstrap.py``), so the public per-provider module set stays identical to RunPod's.

Cost-safety invariant: a rented instance is ALWAYS destroyed — the runner's
``finally``, the onstart's self-destroy backstop, the cancel path, and
``sweep_orphans`` (server startup / post-run) each independently guarantee it.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from autoslm._logging import get_logger
from autoslm.providers.base import GPU_INFO, PollResult, min_cuda_modern, vast_gpu_for_offer
from autoslm.providers.runpod.durable import make_hf_heartbeat_reader
from autoslm.providers.vast import api as vast_api

logger = get_logger(__name__)

# Offer-quality floors (beyond verified+datacenter, which are non-negotiable).
RELIABILITY_FLOOR = float(os.environ.get("AUTOSLM_VAST_MIN_RELIABILITY", "0.95"))
MIN_INET_MBPS = float(os.environ.get("AUTOSLM_VAST_MIN_INET_MBPS", "200"))
# How long an instance may sit in a non-running state (image pull) before we give up.
LOAD_TIMEOUT_S = float(os.environ.get("AUTOSLM_VAST_LOAD_TIMEOUT_S", "900"))
# Boards under-report VRAM vs the class nominal (measured live: L4 23034 MB / 24 GB,
# A40 46068 MB / 48 GB = 0.938 of nominal); the server-side gpu_ram filter gets this
# slack, the class gate stays exact (vast_gpu_for_offer).
_SEARCH_VRAM_SLACK = 0.92

# Minimum disk Vast instances are provisioned with (the bootstrap + worker stack +
# weights need headroom regardless of the spec's request). The offer search MUST use
# this same floor so offers with <60 GB disk don't pass the search and then get
# rejected at create time (``create_instance`` enforces the same max).
MIN_DISK_GB = 60.0


def _effective_disk_gb(spec) -> float:
    """The disk size an instance is actually provisioned with (the create-time floor).

    Both the offer search and ``create_instance`` must agree on this, or offers with a
    disk between ``spec.gpu.disk_gb`` and the floor pass the search then fail to rent.
    """
    return max(float(spec.gpu.disk_gb), MIN_DISK_GB)


# Worker image: torch 2.10 cu128 matches WORKER_DEPS's pin and, critically,
# ships the CUDA 12.8 runtime libs the PyPI wheels link against (verified live: the
# cuda13.0 image broke vllm with "libcudart.so.12: cannot open shared object file").
# Blackwell's CUDA-13 requirement is about the host DRIVER (PTX JIT), enforced by
# the ``cuda_max_good`` offer filter — not the image. -devel ships gcc/nvcc for
# Triton (covers WORKER_SYSTEM_DEPS).
DEFAULT_IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel"


@dataclass(frozen=True)
class VastOffer:
    """A normalized, fully-vetted offer (passed every ``usable_offers`` filter)."""

    offer_id: int
    machine_id: int
    gpu: str  # canonical class name (GPU_INFO key)
    vram_gb: int
    dph_total: float
    cuda_max_good: float
    disk_space: float
    reliability: float
    inet_down: float
    geolocation: str


def usable_offers(
    min_vram_gb: int,
    disk_gb: float,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
) -> list[VastOffer]:
    """Verified-datacenter offers able to run the job, cheapest first.

    Server-side filters do the heavy lifting; everything load-bearing is re-checked
    client-side (belt and suspenders — the result rows carry the proof fields).
    """
    rows = vast_api.search_offers(
        int(min_vram_gb * 1024 * _SEARCH_VRAM_SLACK),
        min_disk_gb=disk_gb,
        min_reliability=RELIABILITY_FLOOR,
    )
    max_dph = float(os.environ.get("AUTOSLM_VAST_MAX_DPH", "0") or 0)
    out: list[VastOffer] = []
    for r in rows:
        gpu = vast_gpu_for_offer(str(r.get("gpu_name") or ""), float(r.get("gpu_ram") or 0))
        if gpu is None:  # not a managed class (Ampere+ floor)
            continue
        info = GPU_INFO[gpu]
        dph = float(r.get("dph_total") or 0)
        cuda = float(r.get("cuda_max_good") or 0)
        if (
            r.get("hosting_type") != 1  # datacenter (the result field `datacenter` is null)
            or r.get("verification") != "verified"
            or info.vram_gb < min_vram_gb
            or float(r.get("reliability2") or 0) < RELIABILITY_FLOOR
            or float(r.get("disk_space") or 0) < float(disk_gb)
            or float(r.get("inet_down") or 0) < MIN_INET_MBPS
            or cuda < float(min_cuda_modern(gpu))  # Blackwell needs CUDA-13 drivers
            or dph <= 0
            or (max_dph and dph > max_dph)
            or int(r.get("machine_id") or 0) in exclude_machine_ids
        ):
            continue
        out.append(
            VastOffer(
                offer_id=int(r["id"]),
                machine_id=int(r.get("machine_id") or 0),
                gpu=gpu,
                vram_gb=info.vram_gb,
                dph_total=dph,
                cuda_max_good=cuda,
                disk_space=float(r.get("disk_space") or 0),
                reliability=float(r.get("reliability2") or 0),
                inet_down=float(r.get("inet_down") or 0),
                geolocation=str(r.get("geolocation") or ""),
            )
        )
    return sorted(out, key=lambda o: (o.dph_total, o.vram_gb))


def vast_image(gpu: str) -> str:
    """Docker image for the worker. AUTOSLM_WORKER_IMAGE (fully baked) wins, then
    AUTOSLM_VAST_IMAGE, then the default cu128 stack image (every class — the
    Blackwell driver floor lives in the offer filter, not the image)."""
    baked = os.environ.get("AUTOSLM_WORKER_IMAGE")
    if baked:
        return baked
    return os.environ.get("AUTOSLM_VAST_IMAGE") or DEFAULT_IMAGE


@dataclass
class VastJobHandle:
    """Persisted in RunStatus.remote so any process can reattach/cancel (cf. JobHandle)."""

    instance_id: int
    offer_id: int
    machine_id: int
    label: str
    gpu: str
    hourly_usd: float
    attempt: int
    started_ts: float

    def to_dict(self) -> dict:
        return {
            "provider": "vast",
            "instance_id": self.instance_id,
            "offer_id": self.offer_id,
            "machine_id": self.machine_id,
            "label": self.label,
            "gpu": self.gpu,
            "hourly_usd": self.hourly_usd,
            "attempt": self.attempt,
            "started_ts": self.started_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VastJobHandle:
        return cls(
            instance_id=int(d["instance_id"]),
            offer_id=int(d.get("offer_id") or 0),
            machine_id=int(d.get("machine_id") or 0),
            label=str(d.get("label") or ""),
            gpu=str(d.get("gpu") or ""),
            hourly_usd=float(d.get("hourly_usd") or 0),
            attempt=int(d.get("attempt") or 0),
            started_ts=float(d.get("started_ts") or 0),
        )


def run_label_prefix(run_id: str) -> str:
    """The prefix EVERY instance label for ``run_id`` starts with.

    ``instance_label`` forces the ``autoslm-`` prefix onto run ids that lack it, so the
    orphan-sweep allowlist must apply the SAME transform: a raw run id (e.g. a
    "fail-fast" test id) would otherwise never match its own ``autoslm-…`` labels and a
    live run's instance could be swept (or fail to be protected)."""
    return f"autoslm-{run_id}" if not run_id.startswith("autoslm-") else run_id


def instance_label(run_id: str, seed: int, attempt: int) -> str:
    """Instance label: run-derived so ``sweep_orphans`` can tell ours from anything
    else on the account. Platform run ids already start with ``autoslm-``; anything else
    (direct-API callers, tests) gets the prefix forced — an instance we rented must NEVER
    be invisible to the orphan sweep."""
    return f"{run_label_prefix(run_id)}-s{seed}-a{attempt}"


def build_payload(spec, seed: int, attempt: int) -> dict:
    """The bootstrap's input — field-compatible with _train_body's, plus the bits the
    instance can't infer (HF prefix for markers, wall cap, attempt)."""
    from autoslm.envs.registry import worker_hub_env_ids, worker_pip_for_env
    from autoslm.providers.runpod.train import build_worker_env

    return {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "env": build_worker_env(spec, seed),
        "extra_pip": list(spec.environment.pip)
        or worker_pip_for_env(spec.environment.id, spec.environment.params),
        "hub_env_ids": worker_hub_env_ids(spec.environment.id, spec.environment.params),
        "hf_prefix": f"{spec.phase}/{spec.run_id}/seed{seed}",
        "max_wall_s": max(60, int(spec.gpu.max_wall_seconds)),
        "attempt": int(attempt),
    }


def build_onstart(payload: dict, install_deps: bool = True) -> str:
    """The instance's onstart script: payload + bootstrap shipped as quoted heredocs.

    Everything dynamic travels base64-encoded inside the script — never interpolated
    into shell syntax and never through Vast's env plumbing — so the job-spec JSON
    (quotes, spaces, anything) survives byte-exact. Secrets-wise the script carries
    the same content as the worker env on RunPod (HF token; never provider keys).

    The bootstrap source is the private ``_bootstrap.py`` sibling — an internal detail
    of this provider, not a public module.
    """
    from autoslm.providers.runpod.train import resolve_worker_deps

    payload_b64 = base64.encodebytes(json.dumps(payload).encode()).decode()
    bootstrap_src = (Path(__file__).parent / "_bootstrap.py").read_text()
    if install_deps:
        deps = " ".join(shlex.quote(d) for d in resolve_worker_deps())
        pip_line = f'"$PYBIN" -m pip install --no-cache-dir {deps}'
    else:
        pip_line = ": # deps baked into the image (AUTOSLM_WORKER_IMAGE)"
    # Verified live: Vast's args-mode wrapper resets PATH, so `python3` resolves to
    # the OS python (Ubuntu 24.04 = PEP 668 externally-managed -> pip refuses), not
    # the image's conda env. Prefer the conda python when present (torch baked in),
    # and let pip install into whichever interpreter won.
    return f"""#!/bin/bash
# AutoSLM vast worker (generated by autoslm.providers.vast.durable.build_onstart)
set -x
export PIP_BREAK_SYSTEM_PACKAGES=1
PYBIN=/opt/conda/bin/python; [ -x "$PYBIN" ] || PYBIN=$(command -v python3)
mkdir -p /root/autoslm
cat > /root/autoslm/payload.b64 <<'AUTOSLM_PAYLOAD_EOF'
{payload_b64}AUTOSLM_PAYLOAD_EOF
base64 -d /root/autoslm/payload.b64 > /root/autoslm/payload.json
cat > /root/autoslm/bootstrap.py <<'AUTOSLM_BOOTSTRAP_EOF'
{bootstrap_src}AUTOSLM_BOOTSTRAP_EOF
# A base worker-stack install failure must STOP the script: continuing into
# bootstrap.py with a partially installed env turns a deterministic dependency
# failure into a later import/model crash (or a missing HF marker if
# huggingface_hub never installed). Hold the box first so the control plane can
# pull the log tail (mirrors the bootstrap-failure path below and the extra-pip
# check=True path). The no-deps branch (":") always succeeds, so this is a no-op there.
{pip_line} || {{ echo "AUTOSLM: base worker dependency install failed" >&2; sleep 600; exit 1; }}
"$PYBIN" /root/autoslm/bootstrap.py
AUTOSLM_RC=$?
# On failure, hold the box for 10 min so the control plane can pull the container
# log tail (the only home of early-bootstrap errors); it destroys us much sooner
# when alive. Success self-destroys immediately.
[ "$AUTOSLM_RC" -ne 0 ] && sleep 600
# Self-destroy backstop (the control plane's destroy is primary). CONTAINER_API_KEY
# is the Vast-injected instance-scoped key — the operator key never ships here.
# python, not curl: the worker image is not guaranteed to carry curl.
"$PYBIN" - <<'AUTOSLM_DESTROY_EOF'
import os, urllib.request
iid, key = os.environ.get("CONTAINER_ID"), os.environ.get("CONTAINER_API_KEY")
if iid and key:
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{{iid}}/",
        method="DELETE",
        headers={{"Authorization": f"Bearer {{key}}"}},
    )
    try:
        urllib.request.urlopen(req, timeout=30)
    except Exception as exc:
        print("self-destroy warn:", exc)
AUTOSLM_DESTROY_EOF
exit $AUTOSLM_RC
"""


def deploy_and_submit(
    spec,
    seed: int,
    offers: list[VastOffer],
    attempt: int = 0,
    log=None,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
) -> VastJobHandle:
    """Rent the cheapest offer that will actually take the job; walk on rejection.

    Offers are a live market — between search and rent the cheapest one is often
    gone. We walk up to 5 ranked offers, then refresh the search once.

    ``exclude_machine_ids`` is the run's blacklist (machines that stalled/failed this
    run earlier). The refresh re-search MUST keep them excluded — otherwise a sick
    machine the orchestrator just blacklisted gets re-selected from the fresh market.
    """

    def say(msg: str):
        if log is not None:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log, flush=True)

    if not offers:
        raise vast_api.VastApiError("no usable vast offers (verified datacenter pool empty)")
    payload = build_payload(spec, seed, attempt)
    label = instance_label(spec.run_id, seed, attempt)
    install_deps = not os.environ.get("AUTOSLM_WORKER_IMAGE")
    tried: list[VastOffer] = []
    candidates = list(offers[:5])
    refreshed = False
    last_err: Exception | None = None
    while candidates:
        offer = candidates.pop(0)
        tried.append(offer)
        onstart = build_onstart(payload, install_deps=install_deps)
        try:
            instance_id = vast_api.create_instance(
                offer.offer_id,
                image=vast_image(offer.gpu),
                disk_gb=_effective_disk_gb(spec),
                env={},
                onstart=onstart,
                label=label,
                runtype="args",
            )
        except vast_api.VastApiError as e:
            last_err = e
            say(f"offer {offer.offer_id} ({offer.gpu} ${offer.dph_total:.2f}/hr) rejected: {e}")
            if not candidates and not refreshed:
                refreshed = True
                # Exclude both the machines we just tried this attempt AND the run's
                # standing blacklist (machines that stalled/failed earlier attempts) —
                # otherwise the fresh search can re-select a sick machine the
                # orchestrator deliberately excluded.
                taken = {o.machine_id for o in tried} | set(exclude_machine_ids)
                # Stay within the allocator-approved class pool: the original `offers`
                # are already filtered to the allocated/pinned + validated classes, so
                # the refresh must not widen to any usable offer (which could rent a
                # different or unvalidated GPU than the run spec assumes).
                allowed = {o.gpu for o in offers}
                candidates = [
                    o
                    for o in usable_offers(
                        min(o.vram_gb for o in offers),
                        _effective_disk_gb(spec),
                        exclude_machine_ids=taken,
                    )
                    if o.gpu in allowed
                ][:5]
            continue
        say(
            f"rented vast instance {instance_id}: {offer.gpu} ${offer.dph_total:.2f}/hr "
            f"(offer {offer.offer_id}, {offer.geolocation}, reliability "
            f"{offer.reliability:.3f}) attempt={attempt} seed={seed}"
        )
        return VastJobHandle(
            instance_id=instance_id,
            offer_id=offer.offer_id,
            machine_id=offer.machine_id,
            label=label,
            gpu=offer.gpu,
            hourly_usd=offer.dph_total,
            attempt=attempt,
            started_ts=time.time(),
        )
    raise vast_api.VastApiError(f"all {len(tried)} vast offers rejected the job: {last_err}")


def _make_hf_file_reader(hf_repo: str, path_in_repo: str, min_interval_s: float = 45.0):
    """Rate-limited reader for one HF artifact's text content (None until it exists)."""
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
                token=os.environ.get("HUGGINGFACE_TOKEN"),
                force_download=True,
            )
            with open(p) as f:
                return f.read()
        except Exception:
            return None

    return read


def _failure_detail(
    hf_repo: str, prefix: str, phase: str, marker: dict | None, instance_id: int | None = None
) -> str:
    """Best root-cause detail we can assemble from the HF artifacts."""
    parts = []
    if marker and marker.get("error"):
        parts.append(str(marker["error"]))
    for mode in (phase,):
        content = _make_hf_file_reader(hf_repo, f"{prefix}/error_{mode}.txt")(force=True)
        if content:
            parts.append(f"--- error_{mode}.txt ---\n{content[-2000:]}")
            break
    if instance_id:
        # Early-bootstrap failures (pip/env errors before the worker can reach HF)
        # only ever appear on the container console.
        logs = vast_api.instance_logs(int(instance_id))
        if logs:
            parts.append(f"--- instance log tail ---\n{logs[-3000:]}")
    return "\n".join(parts) or "vast worker terminated without a DONE sentinel"


# Vast instance states that mean "the container is gone / will not progress".
_DEAD_STATES = {"exited", "stopped", "offline", "deleted"}


def poll_vast_job(
    handle: VastJobHandle,
    spec,
    seed: int,
    log=None,
    interval_s: float = 15.0,
    heartbeat_reader=None,
    stall_after_s: float = 1500.0,
    deadline_s: float | None = None,
) -> PollResult:
    """Poll instance status + HF artifacts to a terminal state (cf. durable.poll_job).

    COMPLETED  fresh DONE sentinel on HF -> metrics.json (cost stamped from the
               offer's real $/hr).
    FAILED     attempt marker with ok=false, or instance dead without DONE.
    STALLED    never left loading within LOAD_TIMEOUT_S, heartbeat frozen for
               stall_after_s, or the client-side deadline passed.
    """

    def say(msg: str):
        if log is not None:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=log, flush=True)

    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
    done_reader = _make_hf_file_reader(hf_repo, f"{prefix}/DONE")
    marker_reader = _make_hf_file_reader(
        hf_repo, f"{prefix}/vast_attempt{handle.attempt}.json", min_interval_s=60.0
    )
    metrics_reader = _make_hf_file_reader(hf_repo, f"{prefix}/metrics.json")

    def finish_ok(done_content: str | None = None) -> PollResult:
        raw = metrics_reader(force=True)
        if raw is None:
            return PollResult(False, failure="job_failed", detail="DONE without metrics.json")
        metrics = json.loads(raw)
        # Bill to the worker's completion time, not now: on recovery the control plane
        # may call this hours after the instance wrote DONE and self-destroyed, so
        # time.time() would add the downtime. DONE carries the worker's time.time().
        end_ts = time.time()
        if done_content:
            try:
                done_ts = float(done_content.strip())
                if handle.started_ts <= done_ts <= end_ts:
                    end_ts = done_ts
            except ValueError:
                # Malformed DONE timestamp: keep end_ts = now rather than trusting garbage.
                pass
        wall_h = (end_ts - handle.started_ts) / 3600.0
        metrics["cost_usd"] = round(wall_h * handle.hourly_usd, 6)
        notes = metrics.get("notes") if isinstance(metrics.get("notes"), dict) else {}
        notes.update(
            {
                "provider": "vast",
                "vast_rate_usd_hr": handle.hourly_usd,
                "vast_gpu": handle.gpu,
                "vast_offer_id": handle.offer_id,
            }
        )
        metrics["notes"] = notes
        return PollResult(True, metrics=metrics)

    def done_is_fresh(content: str) -> bool:
        # DONE carries the worker's time.time(); 120 s of clock-skew grace. Anything
        # older predates this attempt (leftover from a prior attempt's resume).
        try:
            return float(content.strip()) > handle.started_ts - 120.0
        except ValueError:
            return False

    start = time.time()
    last_status = None
    last_hb_key = None
    last_progress = time.time()
    became_running = False
    consecutive_poll_errors = 0
    missing_streak = 0
    while True:
        if deadline_s is not None and time.time() - start > deadline_s:
            return PollResult(False, failure="stalled", detail="client-side deadline exceeded")
        try:
            inst = vast_api.get_instance(handle.instance_id)
            consecutive_poll_errors = 0
        except vast_api.VastApiError as e:
            consecutive_poll_errors += 1
            say(f"poll error ({consecutive_poll_errors}): {e}")
            if consecutive_poll_errors >= 8:
                return PollResult(False, failure="poll_error", detail=str(e))
            time.sleep(min(60, interval_s * consecutive_poll_errors))
            continue
        # Verified live: the instance-detail route TRANSIENTLY answers
        # {"instances": null} for perfectly healthy instances (and for brand-new ones
        # before they materialize). A single missing read means nothing — only a
        # sustained streak is a real disappearance.
        missing_streak = missing_streak + 1 if inst is None else 0

        status = (inst or {}).get("actual_status") or ("missing" if inst is None else "unknown")
        if status != last_status:
            say(f"instance {handle.instance_id}: {status}")
            last_status = status
            last_progress = time.time()
        if status == "running":
            became_running = True

        done = done_reader()
        if done is not None and done_is_fresh(done):
            return finish_ok(done)

        if missing_streak >= 4 or status in _DEAD_STATES:
            # One forced final read: the worker may have finished right before the
            # instance self-destroyed (the normal success order on this substrate).
            done = done_reader(force=True)
            if done is not None and done_is_fresh(done):
                return finish_ok(done)
            raw_marker = marker_reader(force=True)
            marker = None
            if raw_marker:
                with contextlib.suppress(ValueError):
                    marker = json.loads(raw_marker)
            return PollResult(
                False,
                failure="job_failed",
                detail=_failure_detail(hf_repo, prefix, spec.phase, marker, handle.instance_id),
            )

        raw_marker = marker_reader()
        if raw_marker:
            try:
                marker = json.loads(raw_marker)
            except ValueError:
                marker = None
            if marker and not marker.get("ok"):
                return PollResult(
                    False,
                    failure="job_failed",
                    detail=_failure_detail(hf_repo, prefix, spec.phase, marker, handle.instance_id),
                )
            if marker and marker.get("ok"):
                done = done_reader(force=True)
                if done is not None and done_is_fresh(done):
                    return finish_ok(done)

        if not became_running and time.time() - start > LOAD_TIMEOUT_S:
            return PollResult(
                False,
                failure="stalled",
                detail=f"instance stuck in '{status}' for {int(time.time() - start)}s "
                f"(image pull / host issue)",
            )

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
                    stage, step, reward = hb.get("stage"), hb.get("step"), hb.get("reward")
                    say(
                        f"worker: stage={stage}"
                        + (f" step={step}" if step is not None else "")
                        + (f" reward={reward:.3f}" if isinstance(reward, int | float) else "")
                    )
        if became_running and time.time() - last_progress > stall_after_s:
            return PollResult(
                False,
                failure="stalled",
                detail=f"no worker progress for {int(time.time() - last_progress)}s "
                f"(instance status {status})",
            )
        time.sleep(interval_s)


def submit_train_durable_vast(
    spec,
    seed: int,
    log=None,
    on_handle=None,
    attempt: int = 0,
    offers: list[VastOffer] | None = None,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
) -> PollResult:
    """Vast equivalent of ``runpod.durable.submit_train_durable``: rent, persist, poll.

    The ``finally`` destroy is the cost-safety primary: every exit path — success,
    failure, stall, exception, KeyboardInterrupt — tears the paid instance down.
    """
    if offers is None:
        info = GPU_INFO[spec.gpu.type]
        offers = [
            o
            for o in usable_offers(
                info.vram_gb,
                _effective_disk_gb(spec),
                exclude_machine_ids=exclude_machine_ids,
            )
            if o.gpu == spec.gpu.type
        ]
    handle = deploy_and_submit(
        spec, seed, offers, attempt=attempt, log=log, exclude_machine_ids=exclude_machine_ids
    )
    # The instance is rented and billing the MOMENT deploy_and_submit returns; the
    # teardown ``finally`` must guard EVERYTHING after that point — including
    # ``on_handle`` (persisting the remote handle can itself raise). Entering the try
    # before on_handle guarantees the paid instance is destroyed even if the handle is
    # never persisted, closing the rent->persist crash window's billing leak.
    try:
        if on_handle is not None:
            on_handle(handle.to_dict())
        hf_repo = spec.train.hf_repo
        prefix = f"{spec.phase}/{spec.run_id}/seed{seed}"
        reader = make_hf_heartbeat_reader(hf_repo, prefix) if hf_repo else None
        stall = float(os.environ.get("AUTOSLM_STALL_AFTER_S", "1500"))
        # Wall cap + provision/install grace; Vast has no server-side execution
        # timeout, so the client deadline (and the bootstrap's own cap) bound spend.
        deadline = max(60, int(spec.gpu.max_wall_seconds)) + 1800
        return poll_vast_job(
            handle,
            spec,
            seed,
            log=log,
            heartbeat_reader=reader,
            stall_after_s=stall,
            deadline_s=deadline,
        )
    finally:
        vast_api.destroy_instance(handle.instance_id)


def cancel(remote: dict) -> None:
    """Cross-process cancel: destroy the persisted instance (stops billing)."""
    instance_id = remote.get("instance_id")
    if instance_id:
        vast_api.destroy_instance(int(instance_id))


def destroy_run_instances(run_id: str) -> list[int]:
    """Destroy every instance belonging to ONE run (labels start with its run id).

    Cancel/GC path: unlike ``sweep_orphans`` this never looks at other runs, so it
    is safe to call while they are in flight. Best-effort: never raises.
    """
    destroyed: list[int] = []
    if not run_id:
        return destroyed
    try:
        instances = vast_api.list_instances()
    except Exception:
        return destroyed
    prefixes = (run_id, f"autoslm-{run_id}")  # instance_label may force the prefix
    for inst in instances:
        iid = inst.get("id")
        if (
            iid
            and str(inst.get("label") or "").startswith(prefixes)
            and vast_api.destroy_instance(int(iid))
        ):
            destroyed.append(int(iid))
    return destroyed


def sweep_orphans(active_labels: set[str] | None = None) -> list[int]:
    """Destroy AutoSLM-labeled instances that no live run owns; return destroyed ids.

    Run at server startup (crash recovery) and after runs (belt and suspenders).
    Only labels carrying the run-id prefix are ever touched — nothing else on the
    account is ours to destroy. Best-effort: never raises.

    ``active_labels`` may be RAW run ids (what the server tracks) — each is passed
    through ``run_label_prefix`` so it matches the SAME forced-``autoslm-`` prefix the
    instance labels carry. Passing an already-prefixed label is fine (idempotent), so a
    live run whose id lacks the prefix is still correctly protected.
    """
    destroyed: list[int] = []
    try:
        instances = vast_api.list_instances()
    except Exception as exc:
        logger.warning("vast orphan sweep skipped: %s", exc)
        return destroyed
    active = {run_label_prefix(a) for a in (active_labels or set())}
    for inst in instances:
        label = str(inst.get("label") or "")
        if not label.startswith("autoslm-"):
            continue
        if any(label.startswith(a) for a in active):
            continue
        iid = inst.get("id")
        if iid and vast_api.destroy_instance(int(iid)):
            destroyed.append(int(iid))
            logger.warning("destroyed orphaned vast instance %s (label %s)", iid, label)
    return destroyed
