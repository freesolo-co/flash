"""Strict, durable identity for a Lambda Cloud training attempt."""

import re
from dataclasses import dataclass

from flash.providers._lifecycle.instances.instance import InstanceJobHandle

_INSTANCE_TYPE = re.compile(r"gpu_[1-8]x_[a-z0-9]+(?:_[a-z0-9]+)*")
_REGION = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass
class LambdaJobHandle(InstanceJobHandle):
    instance_type: str
    region: str
    name: str
    run_id: str
    hf_repo: str
    phase: str
    scope: str
    gpu_count: int = 1
    provider = "lambda"

    @staticmethod
    def _coerce_instance_id(raw):
        from flash.providers.lambda_.api import valid_instance_id

        if not valid_instance_id(raw):
            raise ValueError("invalid Lambda instance id")
        return raw

    def _extra_to_dict(self):
        return {
            "instance_type": self.instance_type,
            "region": self.region,
            "name": self.name,
            "run_id": self.run_id,
            "hf_repo": self.hf_repo,
            "phase": self.phase,
            "gpu_count": self.gpu_count,
            "scope": self.scope,
        }

    @staticmethod
    def _extra_from_dict(data):
        fields = {}
        for key in ("instance_type", "region", "name", "run_id", "hf_repo", "phase", "scope"):
            value = data.get(key)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"invalid Lambda {key}")
            fields[key] = value
        if not _INSTANCE_TYPE.fullmatch(fields["instance_type"]):
            raise ValueError("invalid Lambda instance_type")
        if not _REGION.fullmatch(fields["region"]):
            raise ValueError("invalid Lambda region")
        count = data.get("gpu_count")
        if type(count) is not int or count not in (1, 2, 4, 8):
            raise ValueError("invalid Lambda gpu_count")
        if not fields["instance_type"].startswith(f"gpu_{count}x_"):
            raise ValueError("Lambda gpu_count disagrees with its instance type")
        fields["gpu_count"] = count
        return fields


def require_scope(handle):
    from flash.providers.lambda_.config import scope_fingerprint

    if handle.scope != scope_fingerprint():
        raise RuntimeError("Lambda credentials differ from this attempt's owner")
