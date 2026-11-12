"""Strict, durable identity for a Mithril Spot training attempt."""

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
        if not isinstance(raw, str) or not re.fullmatch(r"bid_[A-Za-z0-9]+", raw):
            raise ValueError("invalid Mithril Spot sandbox id")
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
                raise ValueError(f"invalid Mithril Spot {key}")
            fields[key] = value
        count = data.get("gpu_count")
        if type(count) is not int or count not in (1, 8):
            raise ValueError("invalid Mithril Spot gpu_count")
        fields["gpu_count"] = count
        return fields


_NO_SUCH_BID = frozenset({400, 403, 404, 422})


def require_scope(handle):
    from flash.providers.mithril.config import (
        region_scope_fingerprint,
        scope_fingerprint,
        settings,
    )

    if handle.scope == scope_fingerprint():
        return
    # Attempts launched by region-pinned releases carry the retired project+region scope. Verify
    # them against their own bid's region so removing the setting does not orphan in-flight runs.
    from flash.providers.mithril.client import MithrilApiError, call

    project = settings()["project_id"]
    try:
        bid = call("GET", f"/spot/bids/{handle.instance_id}")
    except MithrilApiError as exc:
        if exc.status not in _NO_SUCH_BID:
            raise  # timeout, 429 or 5xx: ownership is unverified, not disproven; callers retry
        bid = None
    if (
        isinstance(bid, dict)
        and bid.get("project") == project
        and isinstance(bid.get("region"), str)
        and handle.scope == region_scope_fingerprint(project, bid["region"])
    ):
        return
    raise RuntimeError("Mithril Spot credentials or environment differ from this attempt's owner")
