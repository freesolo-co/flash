"""Lambda Cloud run lifecycle building blocks. MUST NOT import the ``jobs`` package ``__init__``."""

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
    label_matches_run,
    run_label_prefix,
)

__all__ = [
    "LambdaInstance",
    "LambdaJobHandle",
    "build_payload",
    "build_user_data",
    "instance_label",
    "label_matches_run",
    "lambda_image",
    "run_label_prefix",
]


@dataclass(frozen=True)
class LambdaInstance:
    """A launchable (region, instance_type, $/hr) for a managed GPU class."""

    gpu: str
    instance_type: str
    region: str
    vram_gb: int
    price_usd_hr: float


@dataclass
class LambdaJobHandle:
    """Persisted in RunStatus.remote so any process can reattach/cancel (cf. base.JobHandle)."""

    instance_id: str
    instance_type: str
    region: str
    name: str
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
    """Return the worker Docker image for the given GPU class (per-SM tag or base)."""
    from flash.providers._worker import WORKER_IMAGE, worker_image_for_gpu

    return worker_image_for_gpu(gpu, allow_default=True) or WORKER_IMAGE


def build_payload(
    spec, seed: int, attempt: int, runtime_secrets: dict | None = None,
    cache_host_mount: str | None = None,
    mode: str | None = None, models: list | None = None, code_prefix: str | None = None,
) -> dict:
    """Build the Lambda bootstrap payload (arm='lambda')."""
    return _shared_build_payload(
        spec, seed, attempt, arm="lambda", runtime_secrets=runtime_secrets,
        cache_host_mount=cache_host_mount, mode=mode, models=models, code_prefix=code_prefix,
    )


def build_user_data(payload: dict, *, gpu: str | None = None) -> str:
    """The Lambda cloud-init user_data (shared builder, runs the Lambda WORKER_IMAGE)."""
    return _shared_build_user_data(payload, image=lambda_image(gpu))
