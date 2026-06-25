"""Pure, monkeypatch-free building blocks for the Lambda Cloud run lifecycle.

The leaf of the ``flash.providers.lambdalabs.jobs`` package: the persisted/normalized dataclasses
(``LambdaInstance``, ``LambdaJobHandle``) and the pure builders that turn a spec into the
instance's name / payload / cloud-init ``user_data``. None of these read the module-global
constants the lifecycle functions expose for monkeypatching, so they live here, away from the
patched surface in ``__init__``.

This module MUST NOT import the ``jobs`` package ``__init__`` (it is imported BY it).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LambdaInstance:
    """A launchable (region, instance_type, $/hr) for a managed GPU class — the Lambda analog of a
    vetted Vast offer."""

    gpu: str  # canonical class name (GPU_INFO key)
    instance_type: str  # Lambda instance-type name (e.g. "gpu_1x_a10")
    region: str
    vram_gb: int
    price_usd_hr: float


@dataclass
class LambdaJobHandle:
    """Persisted in RunStatus.remote so any process can reattach/cancel (cf. base.JobHandle)."""

    instance_id: str
    instance_type: str
    region: str
    name: str  # the sweep-matchable instance name (run-derived; see ``instance_label``)
    gpu: str
    hourly_usd: float
    attempt: int
    started_ts: float

    def to_dict(self) -> dict:
        return {
            "provider": "lambda",
            "instance_id": self.instance_id,
            "instance_type": self.instance_type,
            "region": self.region,
            "name": self.name,
            "gpu": self.gpu,
            "hourly_usd": self.hourly_usd,
            "attempt": self.attempt,
            "started_ts": self.started_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LambdaJobHandle:
        return cls(
            instance_id=str(d["instance_id"]),
            instance_type=str(d.get("instance_type") or ""),
            region=str(d.get("region") or ""),
            name=str(d.get("name") or ""),
            gpu=str(d.get("gpu") or ""),
            hourly_usd=float(d.get("hourly_usd") or 0),
            attempt=int(d.get("attempt") or 0),
            started_ts=float(d.get("started_ts") or 0),
        )


def lambda_image() -> str:
    """Docker image the cloud-init runs on the Lambda host: the prebuilt, PUBLIC ``WORKER_IMAGE``
    (the byte-identical training stack RunPod bakes), so a Lambda run uses the SAME validated
    environment — no per-host dep resolution. ``FLASH_WORKER_IMAGE`` overrides it (e.g. a hotfix)."""
    import os

    from flash.providers.runpod.train import WORKER_IMAGE

    return os.environ.get("FLASH_WORKER_IMAGE") or WORKER_IMAGE


def run_label_prefix(run_id: str) -> str:
    """The prefix EVERY instance name for ``run_id`` starts with.

    ``instance_label`` forces the ``flash-`` prefix onto run ids that lack it, so the orphan-sweep
    allowlist must apply the SAME transform: a raw run id would otherwise never match its own
    ``flash-…`` names and a live run's instance could be swept (or fail to be protected)."""
    return f"flash-{run_id}" if not run_id.startswith("flash-") else run_id


def instance_label(run_id: str, seed: int, attempt: int) -> str:
    """Instance name: run-derived so ``sweep_orphans`` can tell ours from anything else on the
    account. Platform run ids already start with ``flash-``; anything else (direct-API callers,
    tests) gets the prefix forced — an instance we launched must NEVER be invisible to the sweep.
    (Lambda caps ``name`` at 64 chars; platform run ids keep the full label well under that.)"""
    return f"{run_label_prefix(run_id)}-s{seed}-a{attempt}"


def build_payload(spec, seed: int, attempt: int, runtime_secrets: dict | None = None) -> dict:
    """The bootstrap's input — field-compatible with the RunPod ``_train_body`` payload, plus the
    bits the instance can't infer (HF prefix for markers, wall cap, attempt)."""
    from flash.envs.registry import worker_pip_for_env
    from flash.providers.runpod.train import build_worker_env, chalk_extra_pip

    return {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "env": build_worker_env(spec, seed, runtime_secrets=runtime_secrets),
        # The bootstrap pip-installs extra_pip for every job, so the per-run env wheel + the opt-in
        # chalk spec ride along here to reach default runs (mirrors runpod/jobs.submit_run).
        "extra_pip": (list(spec.environment.pip) or worker_pip_for_env(spec.environment.id))
        + chalk_extra_pip(spec),
        "hf_prefix": f"{spec.phase}/{spec.run_id}/seed{seed}",
        "max_wall_s": max(60, int(spec.gpu.max_wall_seconds)),
        "attempt": int(attempt),
    }


# Host helper: best-effort upload of the consolidated boot log to HF. Lambda exposes NO console/log
# API (unlike Vast), so the box pushes its own boot log to HF — the only window into a failure that
# happens BEFORE the worker container can write its own artifacts (docker/GPU not ready, image pull
# failure). Reads creds from the on-box payload.json. Never raises.
_HOSTLOG_PY = """\
import json
try:
    p = json.load(open("/opt/flash/payload.json"))
    from huggingface_hub import HfApi
    HfApi(token=(p.get("env") or {}).get("HF_TOKEN")).upload_file(
        path_or_fileobj="/opt/flash/host_boot.log",
        path_in_repo=p["hf_prefix"] + "/lambda_boot.log",
        repo_id=p["hf_repo"],
        repo_type="dataset",
    )
except Exception:
    pass
"""


def build_user_data(payload: dict) -> str:
    """Cloud-init ``user_data``: run the worker ``WORKER_IMAGE`` via Docker on the Lambda host.

    cloud-init runs this once at first boot as root. Everything dynamic travels base64-encoded
    inside the script (never interpolated into shell syntax), so the job-spec JSON survives
    byte-exact. The full training stack is baked into the image, so the box only needs Docker + an
    NVIDIA GPU — both shipped by the default Lambda Stack image — and the container does the rest
    (fetch code from HF, run the worker, stream artifacts back to HF).

    Secrets-wise the script carries the same content as the worker env on RunPod (HF token, env
    secrets) — the same trust posture as Vast shipping run secrets to a verified-datacenter box.
    The operator's LAMBDA_API_KEY is NEVER shipped (Lambda has no instance-scoped key; teardown is
    control-plane-side via the runner ``finally`` / poll deadline / ``sweep_orphans``).
    """
    image = lambda_image()
    payload_b64 = base64.encodebytes(json.dumps(payload).encode()).decode()
    bootstrap_src = (Path(__file__).parent.parent / "_bootstrap.py").read_text()
    return f"""#!/bin/bash
# Flash lambda worker (generated by flash.providers.lambdalabs.jobs.builders.build_user_data)
set -x
mkdir -p /opt/flash
# Consolidate ALL boot output (this script + the container) into one host log the uploader ships
# to HF — Lambda has no console API, so this is the only window into a pre-worker failure.
exec >>/opt/flash/host_boot.log 2>&1
cat > /opt/flash/payload.b64 <<'FLASH_PAYLOAD_EOF'
{payload_b64}FLASH_PAYLOAD_EOF
base64 -d /opt/flash/payload.b64 > /opt/flash/payload.json
cat > /opt/flash/bootstrap.py <<'FLASH_BOOTSTRAP_EOF'
{bootstrap_src}FLASH_BOOTSTRAP_EOF
cat > /opt/flash/hostlog.py <<'FLASH_HOSTLOG_EOF'
{_HOSTLOG_PY}FLASH_HOSTLOG_EOF
IMAGE={image!r}
# Best-effort host->HF boot-log uploader (detached so it survives cloud-init exiting).
( pip3 install -q huggingface_hub >/dev/null 2>&1 \\
    || python3 -m pip install -q --break-system-packages huggingface_hub >/dev/null 2>&1 || true
  while true; do python3 /opt/flash/hostlog.py >/dev/null 2>&1 || true; sleep 30; done ) &
disown || true
# Lambda Stack ships Docker + the NVIDIA Container Toolkit, but cloud-init can run before they
# finish initializing — wait for both (up to ~8 min) before launching the worker.
for i in $(seq 1 80); do
  if docker info >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then break; fi
  echo "FLASH: waiting for docker+gpu ($i)"; sleep 6
done
docker info >/dev/null 2>&1 || {{ echo "FLASH: docker never became ready" >&2; exit 1; }}
nvidia-smi >/dev/null 2>&1 || {{ echo "FLASH: gpu never became ready" >&2; exit 1; }}
# Pull with retries (the image is large; a transient registry blip must not fail the run).
for i in 1 2 3 4 5; do docker pull "$IMAGE" && break; echo "FLASH: pull retry $i"; sleep 20; done
# Run the worker container detached so cloud-init completes promptly; completion is signaled via
# the worker's HF artifacts (DONE/metrics.json/marker), never a return channel from the box.
docker run -d --name flashrun --gpus all --shm-size=16g --network host \\
  -v /opt/flash:/root/flash -w /root/flash \\
  "$IMAGE" python /root/flash/bootstrap.py
# Mirror the container's stdout into the host boot log (detached) so an early in-container crash is
# visible on HF even if it dies before uploading its own console artifact.
( docker logs -f flashrun >>/opt/flash/host_boot.log 2>&1 || true ) &
disown || true
"""
