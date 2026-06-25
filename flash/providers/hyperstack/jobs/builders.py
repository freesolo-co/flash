"""Pure, monkeypatch-free building blocks for the Hyperstack run lifecycle.

The Hyperstack-specific leaf of ``flash.providers.hyperstack.jobs``: the normalized dataclasses
(``HyperstackInstance``, ``HyperstackJobHandle``) and the image accessor. The cross-provider
pieces (sweep label, bootstrap payload, cloud-init ``user_data``) are shared with Lambda in
``flash.providers._instance`` and re-exported here.

This module MUST NOT import the ``jobs`` package ``__init__`` (it is imported BY it).
"""

from __future__ import annotations

from dataclasses import dataclass

from flash.providers._instance import (
    build_payload as _shared_build_payload,
)
from flash.providers._instance import (
    build_user_data as _shared_build_user_data,
)
from flash.providers._instance import (
    instance_label,
    run_label_prefix,
)

__all__ = [
    "HyperstackInstance",
    "HyperstackJobHandle",
    "build_payload",
    "build_user_data",
    "hyperstack_image",
    "instance_label",
    "run_label_prefix",
]


@dataclass(frozen=True)
class HyperstackInstance:
    """A launchable (region, flavor, $/hr) for a managed GPU class (the Hyperstack analog of a
    Lambda instance candidate)."""

    gpu: str  # canonical class name (GPU_INFO key)
    flavor: str  # Hyperstack flavor name (e.g. "n3-L40x1")
    region: str
    environment: str  # default-<region>
    vram_gb: int
    price_usd_hr: float


@dataclass
class HyperstackJobHandle:
    """Persisted in RunStatus.remote so any process can reattach/cancel (cf. base.JobHandle)."""

    vm_id: str
    flavor: str
    region: str
    name: str  # the sweep-matchable VM name (run-derived; see ``instance_label``)
    gpu: str
    hourly_usd: float
    attempt: int
    started_ts: float

    def to_dict(self) -> dict:
        return {
            "provider": "hyperstack",
            "vm_id": self.vm_id,
            "flavor": self.flavor,
            "region": self.region,
            "name": self.name,
            "gpu": self.gpu,
            "hourly_usd": self.hourly_usd,
            "attempt": self.attempt,
            "started_ts": self.started_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HyperstackJobHandle:
        return cls(
            vm_id=str(d["vm_id"]),
            flavor=str(d.get("flavor") or ""),
            region=str(d.get("region") or ""),
            name=str(d.get("name") or ""),
            gpu=str(d.get("gpu") or ""),
            hourly_usd=float(d.get("hourly_usd") or 0),
            attempt=int(d.get("attempt") or 0),
            started_ts=float(d.get("started_ts") or 0),
        )


def hyperstack_image(gpu: str | None = None) -> str:
    """Docker image the cloud-init runs on the Hyperstack host: the prebuilt, PUBLIC ``WORKER_IMAGE``
    (the byte-identical training stack RunPod bakes). ``FLASH_WORKER_IMAGE`` overrides it; when the
    operator opts into per-SM warmed images (``FLASH_WORKER_IMAGE_PER_SM`` /
    ``FLASH_WORKER_IMAGE_TEMPLATE``), the GPU class selects the matching ``-smXX`` tag so the worker's
    baked kernel cache matches the rented GPU's arch (the same selector RunPod uses). NB: this is the
    *container* image; the Hyperstack VM *boot* image (Docker-preinstalled Ubuntu/CUDA) is chosen
    separately in ``api.docker_image_for_region``."""
    from flash.providers.runpod.train import WORKER_IMAGE, worker_image_for_gpu

    # allow_default=True -> always a concrete image to docker-pull (override / per-sm tag / base).
    return worker_image_for_gpu(gpu, allow_default=True) or WORKER_IMAGE


def build_payload(
    spec, seed: int, attempt: int, runtime_secrets: dict | None = None,
    cache_host_mount: str | None = None, cache_block_device: bool = False,
    mode: str | None = None, models: list | None = None,
) -> dict:
    """The Hyperstack bootstrap payload (shared builder, arm='hyperstack'). ``cache_host_mount`` (the
    host path the attached block volume is formatted+mounted at) points HF_HOME at it;
    ``cache_block_device`` enables the cloud-init wait-for-device/format/mount preamble.
    ``mode='preload'`` + ``models`` makes it a download-only warm payload (no worker)."""
    return _shared_build_payload(
        spec, seed, attempt, arm="hyperstack", runtime_secrets=runtime_secrets,
        cache_host_mount=cache_host_mount, cache_block_device=cache_block_device,
        mode=mode, models=models,
    )


def build_user_data(payload: dict, *, gpu: str | None = None) -> str:
    """The Hyperstack cloud-init user_data (shared builder, runs the worker WORKER_IMAGE)."""
    return _shared_build_user_data(payload, image=hyperstack_image(gpu))
