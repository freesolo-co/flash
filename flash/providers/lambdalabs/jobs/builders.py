"""Lambda Cloud run lifecycle building blocks. MUST NOT import the ``jobs`` package ``__init__``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from flash.providers._instance import (
    InstanceJobHandle,
    instance_label,
    label_matches_run,
    run_label_prefix,
)
from flash.providers._instance import (
    build_payload as _shared_build_payload,
)
from flash.providers._instance import (
    build_user_data as _shared_build_user_data,
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
class LambdaJobHandle(InstanceJobHandle):
    """Persisted in RunStatus.remote so any process can reattach/cancel (cf. base.JobHandle).

    Extends the shared ``InstanceJobHandle`` with Lambda's region/instance-type locator fields; the
    common fields + (de)serialization live on the base.
    """

    instance_type: str
    region: str
    name: str

    provider: ClassVar[str] = "lambda"

    @staticmethod
    def _coerce_instance_id(raw) -> str:
        if not isinstance(raw, str) or not raw:
            raise ValueError("invalid lambda instance id")
        return raw

    def _extra_to_dict(self) -> dict:
        return {
            "instance_type": self.instance_type,
            "region": self.region,
            "name": self.name,
        }

    @staticmethod
    def _extra_from_dict(d: dict) -> dict:
        values = {key: d.get(key) for key in ("instance_type", "region", "name")}
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise ValueError("persisted lambda provider identity is incomplete")
        return values


def lambda_image(gpu: str | None = None) -> str:
    """Return the worker Docker image for the given GPU class (per-SM tag or base)."""
    from flash.providers._worker import WORKER_IMAGE, worker_image_for_gpu

    return worker_image_for_gpu(gpu, allow_default=True) or WORKER_IMAGE


def build_payload(
    spec,
    seed: int,
    attempt: int,
    runtime_secrets: dict | None = None,
    cache_host_mount: str | None = None,
    mode: str | None = None,
    models: list | None = None,
    code_prefix: str | None = None,
    deadline_at: float | None = None,
) -> dict:
    """Build the Lambda bootstrap payload (arm='lambda')."""
    return _shared_build_payload(
        spec,
        seed,
        attempt,
        arm="lambda",
        runtime_secrets=runtime_secrets,
        cache_host_mount=cache_host_mount,
        mode=mode,
        models=models,
        code_prefix=code_prefix,
        deadline_at=deadline_at,
    )


def build_user_data(payload: dict, *, gpu: str | None = None) -> str:
    """The Lambda cloud-init user_data (shared builder, runs the Lambda WORKER_IMAGE)."""
    return _shared_build_user_data(payload, image=lambda_image(gpu))
