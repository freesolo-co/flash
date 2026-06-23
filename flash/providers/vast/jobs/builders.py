"""Pure, monkeypatch-free building blocks for the Vast run lifecycle.

The leaf of the ``flash.providers.vast.jobs`` package: the persisted/normalized
dataclasses (``VastOffer``, ``VastJobHandle``) and the pure builders that turn a spec
into the instance's label / image / payload / onstart script. None of these read the
module-global constants the lifecycle functions expose for monkeypatching
(``LOAD_TIMEOUT_S``/``RELIABILITY_FLOOR``/``MIN_INET_MBPS``/``MIN_DISK_GB``/``time``),
so they live here, away from the patched surface in ``__init__``.

This module MUST NOT import the ``jobs`` package ``__init__`` (it is imported BY it).
"""

from __future__ import annotations

import base64
import json
import shlex
from dataclasses import dataclass
from pathlib import Path


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


def vast_image() -> str:
    """Docker image for the worker: the prebuilt, PUBLIC WORKER_IMAGE (full training stack baked
    in). Used for every GPU class — the Blackwell driver floor lives in the ``cuda_max_good`` offer
    filter, not the image. There is no generic fallback image; the worker image is always used."""
    from flash.providers.runpod.train import WORKER_IMAGE

    return WORKER_IMAGE


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

    ``instance_label`` forces the ``flash-`` prefix onto run ids that lack it, so the
    orphan-sweep allowlist must apply the SAME transform: a raw run id (e.g. a
    "fail-fast" test id) would otherwise never match its own ``flash-…`` labels and a
    live run's instance could be swept (or fail to be protected)."""
    return f"flash-{run_id}" if not run_id.startswith("flash-") else run_id


def instance_label(run_id: str, seed: int, attempt: int) -> str:
    """Instance label: run-derived so ``sweep_orphans`` can tell ours from anything
    else on the account. Platform run ids already start with ``flash-``; anything else
    (direct-API callers, tests) gets the prefix forced — an instance we rented must NEVER
    be invisible to the orphan sweep."""
    return f"{run_label_prefix(run_id)}-s{seed}-a{attempt}"


def build_payload(
    spec,
    seed: int,
    attempt: int,
    runtime_secrets: dict[str, str] | None = None,
) -> dict:
    """The bootstrap's input — field-compatible with _train_body's, plus the bits the
    instance can't infer (HF prefix for markers, wall cap, attempt)."""
    from flash.envs.registry import worker_pip_for_env
    from flash.providers.runpod.train import build_worker_env, chalk_extra_pip

    return {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "env": build_worker_env(spec, seed, runtime_secrets=runtime_secrets),
        # The Vast bootstrap pip-installs extra_pip for every job (provider/vast/_bootstrap.py),
        # so the opt-in chalk spec rides along here to reach default runs — see chalk_extra_pip().
        "extra_pip": (list(spec.environment.pip) or worker_pip_for_env(spec.environment.id))
        + chalk_extra_pip(spec),
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
    from flash.providers.runpod.train import resolve_worker_deps

    payload_b64 = base64.encodebytes(json.dumps(payload).encode()).decode()
    bootstrap_src = (Path(__file__).parent.parent / "_bootstrap.py").read_text()
    if install_deps:
        deps = " ".join(shlex.quote(d) for d in resolve_worker_deps())
        # Vast cold-start is dominated by this dep install (torch/vllm/transformers cu128 stack) on
        # fresh hosts — RunPod caches it as a Flash artifact, but Vast reinstalls per host, so use
        # `uv pip` (validated in the worker image: resolves + installs the full pinned cu128 stack
        # in ~12s, ~10x faster than pip). --break-system-packages: PYBIN is the image's
        # externally-managed system python (newer pytorch images dropped /opt/conda); uv refuses it
        # without this flag (and ignores pip's PIP_BREAK_SYSTEM_PACKAGES), which is what silently
        # broke the earlier uv attempt. The flag is a no-op on a conda/venv python, so it's safe
        # across image variants.
        pip_line = (
            '"$PYBIN" -m pip install --no-cache-dir uv '
            f'&& "$PYBIN" -m uv pip install --python "$PYBIN" --break-system-packages --no-cache {deps}'
        )
    else:
        pip_line = ": # deps baked into the image (WORKER_IMAGE)"
    # Verified live: Vast's args-mode wrapper resets PATH, so `python3` resolves to
    # the OS python (Ubuntu 24.04 = PEP 668 externally-managed -> pip refuses), not
    # the image's conda env. Prefer the conda python when present (torch baked in),
    # and let pip install into whichever interpreter won.
    return f"""#!/bin/bash
# Flash vast worker (generated by flash.providers.vast.jobs.build_onstart)
set -x
export PIP_BREAK_SYSTEM_PACKAGES=1
PYBIN=/opt/conda/bin/python; [ -x "$PYBIN" ] || PYBIN=/usr/local/bin/python; [ -x "$PYBIN" ] || PYBIN=$(command -v python3)
mkdir -p /root/flash
cat > /root/flash/payload.b64 <<'FLASH_PAYLOAD_EOF'
{payload_b64}FLASH_PAYLOAD_EOF
base64 -d /root/flash/payload.b64 > /root/flash/payload.json
cat > /root/flash/bootstrap.py <<'FLASH_BOOTSTRAP_EOF'
{bootstrap_src}FLASH_BOOTSTRAP_EOF
# A base worker-stack install failure must STOP the script: continuing into
# bootstrap.py with a partially installed env turns a deterministic dependency
# failure into a later import/model crash (or a missing HF marker if
# huggingface_hub never installed). Hold the box first so the control plane can
# pull the log tail (mirrors the bootstrap-failure path below and the extra-pip
# check=True path). The no-deps branch (":") always succeeds, so this is a no-op there.
{pip_line} || {{ echo "FLASH: base worker dependency install failed" >&2; sleep 600; exit 1; }}
"$PYBIN" /root/flash/bootstrap.py
FLASH_RC=$?
# On failure, hold the box for 10 min so the control plane can pull the container
# log tail (the only home of early-bootstrap errors); it destroys us much sooner
# when alive. Success self-destroys immediately.
[ "$FLASH_RC" -ne 0 ] && sleep 600
# Self-destroy backstop (the control plane's destroy is primary). CONTAINER_API_KEY
# is the Vast-injected instance-scoped key — the operator key never ships here.
# python, not curl: the worker image is not guaranteed to carry curl.
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
