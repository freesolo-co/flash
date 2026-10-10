"""Pure, monkeypatch-free building blocks for the Vast.ai run lifecycle.

The normalized dataclasses (``VastOffer``, ``VastJobHandle``), the image accessor, and the container
``onstart`` script. Cross-provider pieces (the sweep label, the bootstrap payload) come from the
shared ``flash.providers._lifecycle.instance`` so Vast stays byte-identical to Lambda on substrate-neutral
parts. Vast rents a CONTAINER directly (image + args), not a VM you cloud-init, so there is no
``build_user_data``/``docker run`` — ``build_onstart`` runs the shared bootstrap as the container command.

MUST NOT import the ``jobs`` package ``__init__`` (it is imported BY it).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import ClassVar

from flash.providers._lifecycle.instances.instance import (
    InstanceJobHandle,
    _instance_capsule,
    _spill_large_spec_to_hf,
    instance_label,
    label_matches_run,
    run_label_prefix,
)

# Shared instance-provider helpers (single source of truth; Vast binds arm="vast" + its own onstart).
from flash.providers._lifecycle.instances.instance import (
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
    """A normalized, fully-vetted offer (passed every ``usable_offers`` filter).

    ``vram_gb`` and ``dph_total`` are PER CARD. Vast's raw ``dph_total`` prices the whole offer, so
    ``usable_offers`` divides it by the offer's card count on the way in; the whole-box rate is
    ``gpu_count * dph_total``.
    """

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
    gpu_count: int = 1


@dataclass
class VastJobHandle(InstanceJobHandle):
    """Persisted in RunStatus.remote so any process can reattach/cancel (cf. base.JobHandle).

    Extends the shared ``InstanceJobHandle`` with Vast's offer/machine locator fields; the common
    fields + (de)serialization live on the base.
    """

    offer_id: int
    machine_id: int
    label: str

    provider: ClassVar[str] = "vast"

    @staticmethod
    def _coerce_instance_id(raw) -> int:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError("invalid vast instance id")
        return raw

    def _extra_to_dict(self) -> dict:
        return {
            "offer_id": self.offer_id,
            "machine_id": self.machine_id,
            "label": self.label,
        }

    @staticmethod
    def _extra_from_dict(d: dict) -> dict:
        offer_id = d.get("offer_id")
        machine_id = d.get("machine_id")
        label = d.get("label")
        if (
            isinstance(offer_id, bool)
            or not isinstance(offer_id, int)
            or offer_id <= 0
            or isinstance(machine_id, bool)
            or not isinstance(machine_id, int)
            or machine_id <= 0
            or not isinstance(label, str)
            or not label
        ):
            raise ValueError("persisted vast provider identity is incomplete")
        return {"offer_id": offer_id, "machine_id": machine_id, "label": label}


def vast_image(gpu: str | None = None) -> str:
    """Docker image for the rented container: the prebuilt PUBLIC worker image, routed through
    ``worker_image_for_gpu`` so Vast selects the same per-SM kernel-cache image RunPod and Lambda do.
    Vast runs the worker via its own onstart, so the image's CMD is irrelevant — only the baked
    deps/cache matter. The Blackwell driver floor lives in the ``cuda_max_good`` offer filter, not
    the image."""
    from flash.providers._lifecycle.net.worker import worker_image_for_gpu

    return worker_image_for_gpu(gpu)


def build_payload(
    spec,
    attempt: int,
    runtime_secrets: dict | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> dict:
    """The Vast bootstrap payload (shared builder, arm='vast').

    Vast has no per-region weight-cache filesystem (that is Lambda's NFS feature), so it never
    passes ``cache_host_mount``/``mode`` — a plain cold worker payload every time.
    """
    return _shared_build_payload(
        spec,
        attempt,
        arm="vast",
        runtime_secrets=runtime_secrets,
        source_snapshot=source_snapshot,
        deadline_at=deadline_at,
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
    # Ship the SHARED instance bootstrap and its siblings as ONE verified capsule -- the same
    # artifact Lambda runs, so the two providers cannot drift. Its members are sha256'd in the
    # manifest and the archive is checked against the digest below before anything executes: on a
    # box already rented and billing, a truncated or substituted payload must fail loudly rather
    # than half-install.
    capsule_b64, capsule_sha256 = _instance_capsule()
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
cat > /root/flash/capsule.b64 <<'FLASH_CAPSULE_EOF'
{capsule_b64}
FLASH_CAPSULE_EOF
base64 -d /root/flash/capsule.b64 > /root/flash/capsule.pyz
# verify BEFORE the first execution. the expected digest is supplied by the control plane, not read
# out of the archive, so a consistently-rewritten capsule still fails here.
if ! echo "{capsule_sha256}  /root/flash/capsule.pyz" | sha256sum -c - >/dev/null 2>&1; then
  echo "flash: runtime capsule failed verification" >&2
  FLASH_RC=1
else
"$PYBIN" /root/flash/capsule.pyz bootstrap
FLASH_RC=$?
fi
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
