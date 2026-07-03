"""Pure, monkeypatch-free building blocks for the Vast.ai run lifecycle.

The normalized dataclasses (``VastOffer``, ``VastJobHandle``), the image accessor, and the container
``onstart`` script. Cross-provider pieces (the sweep label, the bootstrap payload) come from the
shared ``flash.providers._instance`` so Vast stays byte-identical to Lambda on substrate-neutral
parts. Vast rents a CONTAINER directly (image + args), not a VM you cloud-init, so there is no
``build_user_data``/``docker run`` — ``build_onstart`` runs the shared bootstrap as the container command.

MUST NOT import the ``jobs`` package ``__init__`` (it is imported BY it).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from flash.providers._instance import (
    _spill_large_spec_to_hf,
    instance_label,
    label_matches_run,
    run_label_prefix,
)

# Shared instance-provider helpers (single source of truth; Vast binds arm="vast" + its own onstart).
from flash.providers._instance import (
    build_payload as _shared_build_payload,
)

__all__ = [
    "VastJobHandle",
    "VastOffer",
    "build_onstart",
    "build_payload",
    "instance_label",
    "label_matches_run",
    "run_label_prefix",
    "vast_image",
]


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


@dataclass
class VastJobHandle:
    """Persisted in RunStatus.remote so any process can reattach/cancel (cf. base.JobHandle)."""

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
        # instance_id identifies the box (poll/destroy target), so unlike the other fields it has no safe
        # default — a 0/None would point teardown at a non-existent instance. A corrupt/partial persisted
        # handle must fail with a clear, actionable error, not a bare KeyError/ValueError.
        try:
            instance_id = int(d["instance_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"corrupt vast handle: missing/non-numeric instance_id "
                f"({d.get('instance_id')!r}) in persisted handle {d!r}"
            ) from exc
        return cls(
            instance_id=instance_id,
            offer_id=int(d.get("offer_id") or 0),
            machine_id=int(d.get("machine_id") or 0),
            label=str(d.get("label") or ""),
            gpu=str(d.get("gpu") or ""),
            hourly_usd=float(d.get("hourly_usd") or 0),
            attempt=int(d.get("attempt") or 0),
            started_ts=float(d.get("started_ts") or 0),
        )


def vast_image(gpu: str | None = None) -> str:
    """Docker image for the rented container: the prebuilt PUBLIC worker image, routed through
    ``worker_image_for_gpu`` so the same operator overrides RunPod/Lambda honor apply to Vast too
    (``FLASH_WORKER_IMAGE`` and the per-SM kernel-cache image). Vast runs the worker via its own onstart,
    so the image's CMD is irrelevant — only the baked deps/cache matter. The Blackwell driver floor lives
    in the ``cuda_max_good`` offer filter, not the image."""
    from flash.providers.runpod.train.deps import WORKER_IMAGE, worker_image_for_gpu

    return worker_image_for_gpu(gpu) or WORKER_IMAGE


def build_payload(
    spec,
    seed: int,
    attempt: int,
    runtime_secrets: dict | None = None,
    code_prefix: str | None = None,
) -> dict:
    """The Vast bootstrap payload (shared builder, arm='vast').

    Vast has no per-region weight-cache filesystem (that is Lambda's NFS feature), so it never
    passes ``cache_host_mount``/``mode`` — a plain cold worker payload every time.
    """
    return _shared_build_payload(
        spec,
        seed,
        attempt,
        arm="vast",
        runtime_secrets=runtime_secrets,
        code_prefix=code_prefix,
    )


def build_onstart(payload: dict) -> str:
    """The rented container's command: ship the payload + shared instance bootstrap as quoted heredocs,
    run it, then self-destroy.

    Vast runs this with ``runtype="args"`` (``bash -c <onstart>``), so the script IS the container command
    and no SSH key is needed on the account. Everything dynamic travels base64-encoded (never interpolated
    into shell syntax), so the job-spec JSON survives byte-exact. The training stack is baked into the
    worker image, so only the shared bootstrap runs here — installs the per-run ``extra_pip``, fetches the
    flash code from HF, runs the worker, uploads the ``vast_attempt<N>.json`` marker the poller keys on. The
    bootstrap is the SHARED ``_instance_bootstrap.py`` Lambda also runs, so in-container behavior is
    identical across substrates.
    """
    # Spill a large job spec to HF first (like Lambda's build_user_data): a big inline spec balloons the
    # base64 payload and can blow Vast's onstart length limit, failing the rent. Idempotent.
    payload = _spill_large_spec_to_hf(payload)
    payload_b64 = base64.encodebytes(json.dumps(payload).encode()).decode()
    # Ship the SHARED instance bootstrap (sibling of the vast package's parent: providers/_instance_bootstrap.py).
    bootstrap_src = (Path(__file__).parent.parent.parent / "_instance_bootstrap.py").read_text()
    # Vast's args-mode wrapper resets PATH, so `python3` can resolve to the OS python (PEP 668
    # externally-managed), not the image's stack python. Prefer the image's baked interpreter
    # (conda / /usr/local) where torch + huggingface_hub live; fall back to python3.
    return f"""#!/bin/bash
# Flash vast worker (generated by flash.providers.vast.jobs.build_onstart; arm={payload.get("flash_arm")})
set -x
export PIP_BREAK_SYSTEM_PACKAGES=1
PYBIN=/opt/conda/bin/python; [ -x "$PYBIN" ] || PYBIN=/usr/local/bin/python; [ -x "$PYBIN" ] || PYBIN=$(command -v python3 || command -v python)
# No python at all: neither the bootstrap NOR the python-based self-destroy backstop below can run, so
# don't fall through to confusing "command not found" failures. Hold briefly so the control plane can
# pull the log tail, then exit non-zero — the instance carries the flash- label, so the poller's
# stall/first-liveness detection + sweep_orphans reap it (the control-plane destroy is primary anyway).
if [ -z "$PYBIN" ]; then
  echo "flash: no python interpreter (python3/python) on PATH or at /opt/conda,/usr/local; cannot run bootstrap or self-destroy" >&2
  sleep 600
  exit 1
fi
mkdir -p /root/flash
cat > /root/flash/payload.b64 <<'FLASH_PAYLOAD_EOF'
{payload_b64}
FLASH_PAYLOAD_EOF
base64 -d /root/flash/payload.b64 > /root/flash/payload.json
cat > /root/flash/bootstrap.py <<'FLASH_BOOTSTRAP_EOF'
{bootstrap_src}
FLASH_BOOTSTRAP_EOF
"$PYBIN" /root/flash/bootstrap.py
FLASH_RC=$?
# On failure, hold the box for 10 min so the control plane can pull the container log tail via the
# Vast logs API (the only home of early-bootstrap errors visible before the worker reaches HF); it
# destroys us much sooner when alive. Success self-destroys immediately.
[ "$FLASH_RC" -ne 0 ] && sleep 600
# Self-destroy backstop (the control plane's destroy is primary). CONTAINER_API_KEY is the Vast-
# injected instance-scoped key — the operator key never ships here. python, not curl: the worker
# image is not guaranteed to carry curl.
"$PYBIN" - <<'FLASH_DESTROY_EOF'
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
FLASH_DESTROY_EOF
exit $FLASH_RC
"""
