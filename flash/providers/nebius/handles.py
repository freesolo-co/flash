"""Strict, durable identity for a Nebius Spot training attempt."""

import re
from dataclasses import dataclass

from flash.providers._lifecycle.instances.instance import InstanceJobHandle


@dataclass
class SpotJobHandle(InstanceJobHandle):
    name: str
    run_id: str
    hf_repo: str
    phase: str
    scope: str
    gpu_count: int = 1
    provider = "spot"

    @staticmethod
    def _coerce_instance_id(raw):
        if not isinstance(raw, str) or not re.fullmatch(r"computeinstance-[A-Za-z0-9_-]+", raw):
            raise ValueError("invalid Nebius Spot sandbox id")
        return raw

    def _extra_to_dict(self):
        return {
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
        for key in ("name", "run_id", "hf_repo", "phase", "scope"):
            value = data.get(key)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"invalid Nebius Spot {key}")
            fields[key] = value
        count = data.get("gpu_count")
        if type(count) is not int or count not in (1, 8):
            raise ValueError("invalid Nebius Spot gpu_count")
        fields["gpu_count"] = count
        return fields


def require_scope(handle):
    from flash.providers.nebius.config import scope_fingerprint

    if handle.scope != scope_fingerprint():
        raise RuntimeError(
            "Nebius Spot credentials or environment differ from this attempt's owner"
        )
