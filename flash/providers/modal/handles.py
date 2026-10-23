"""Strict, durable identity for a Modal training attempt."""

import math
import re
from dataclasses import dataclass, field

from flash.providers._lifecycle.instances.instance import InstanceJobHandle


@dataclass
class ModalJobHandle(InstanceJobHandle):
    name: str
    run_id: str
    hf_repo: str
    phase: str
    scope: str
    gpu_count: int = 1
    cpu_uploads: bool = False
    cpu_prefetch: dict = field(default_factory=dict)
    provider = "modal"

    @staticmethod
    def _coerce_instance_id(raw):
        if not isinstance(raw, str) or not re.fullmatch(r"sb-[A-Za-z0-9_-]+", raw):
            raise ValueError("invalid Modal sandbox id")
        return raw

    def _extra_to_dict(self):
        return {
            "name": self.name,
            "run_id": self.run_id,
            "hf_repo": self.hf_repo,
            "phase": self.phase,
            "gpu_count": self.gpu_count,
            "scope": self.scope,
            **({"cpu_uploads": True} if self.cpu_uploads else {}),
            **({"cpu_prefetch": self.cpu_prefetch} if self.cpu_prefetch else {}),
        }

    @staticmethod
    def _extra_from_dict(data):
        fields = {}
        for key in ("name", "run_id", "hf_repo", "phase", "scope"):
            value = data.get(key)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"invalid Modal {key}")
            fields[key] = value
        count = data.get("gpu_count")
        if type(count) is not int or count not in (1, 2, 4, 8):
            raise ValueError("invalid Modal gpu_count")
        fields["gpu_count"] = count
        flag = data.get("cpu_uploads", False)
        if type(flag) is not bool:
            raise ValueError("invalid Modal cpu_uploads")
        fields["cpu_uploads"] = flag
        prefetch = data.get("cpu_prefetch", {})
        if not isinstance(prefetch, dict):
            raise ValueError("invalid Modal cpu_prefetch")
        if prefetch:
            if set(prefetch) != {"sandbox_id", "wall_seconds", "cost_upper_estimate_usd"}:
                raise ValueError("invalid Modal cpu_prefetch fields")
            ModalJobHandle._coerce_instance_id(prefetch["sandbox_id"])
            for key in ("wall_seconds", "cost_upper_estimate_usd"):
                value = prefetch[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid Modal cpu_prefetch cost")
        fields["cpu_prefetch"] = dict(prefetch)
        return fields


def require_scope(handle):
    from flash.providers.modal.auth import scope_fingerprint

    if handle.scope != scope_fingerprint():
        raise RuntimeError("Modal credentials or environment differ from this attempt's owner")
