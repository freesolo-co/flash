"""Pure, monkeypatch-free building blocks for the Lambda Cloud run lifecycle.

The Lambda-specific leaf of ``flash.providers.lambdalabs.jobs``: the normalized dataclasses
(``LambdaInstance``, ``LambdaJobHandle``) and the image accessor. The cross-provider pieces — the
run-derived sweep label, the bootstrap payload, and the cloud-init ``user_data`` — live in the
shared ``flash.providers._instance`` and are re-exported here so the import path is unchanged.

This module MUST NOT import the ``jobs`` package ``__init__`` (it is imported BY it).
"""

from __future__ import annotations

from dataclasses import dataclass

# Shared instance-provider helpers (single source of truth; Lambda binds arm="lambda" + its image).
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
    "LambdaInstance",
    "LambdaJobHandle",
    "build_payload",
    "build_user_data",
    "instance_label",
    "lambda_image",
    "run_label_prefix",
]


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


def lambda_image(gpu: str | None = None) -> str:
    """Docker image the cloud-init runs on the Lambda host: the prebuilt, PUBLIC ``WORKER_IMAGE``
    (the byte-identical training stack RunPod bakes). ``FLASH_WORKER_IMAGE`` overrides it; when the
    operator opts into per-SM warmed images (``FLASH_WORKER_IMAGE_PER_SM`` /
    ``FLASH_WORKER_IMAGE_TEMPLATE``), the GPU class selects the matching ``-smXX`` tag so the worker's
    baked kernel cache matches the rented GPU's arch (the same selector RunPod uses)."""
    from flash.providers.runpod.train import WORKER_IMAGE, worker_image_for_gpu

    # allow_default=True -> always a concrete image to docker-pull (override / per-sm tag / base).
    return worker_image_for_gpu(gpu, allow_default=True) or WORKER_IMAGE


def build_payload(
    spec, seed: int, attempt: int, runtime_secrets: dict | None = None,
    cache_host_mount: str | None = None,
    mode: str | None = None, models: list | None = None,
) -> dict:
    """The Lambda bootstrap payload (shared builder, arm='lambda'). ``cache_host_mount`` (the host
    NFS mount of the attached weight-cache filesystem, /lambda/nfs/<name>) points the base-model
    prefetch (FLASH_WEIGHT_CACHE_DIR) at it.
    ``mode='preload'`` + ``models`` makes it a download-only warm payload (no worker)."""
    return _shared_build_payload(
        spec, seed, attempt, arm="lambda", runtime_secrets=runtime_secrets,
        cache_host_mount=cache_host_mount, mode=mode, models=models,
    )


def build_user_data(payload: dict, *, gpu: str | None = None) -> str:
    """The Lambda cloud-init user_data (shared builder, runs the Lambda WORKER_IMAGE)."""
    return _shared_build_user_data(payload, image=lambda_image(gpu))
