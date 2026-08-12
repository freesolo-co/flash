"""Cost accounting + the standard run-metrics record for Flash runs.

Customer GPU cost = training-loop GPU hours * hourly_rate; setup/cold-start time is reported
separately for observability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

_PRIVATE_SOURCE_MARKER = "[prepared warm-start source]"
_PRIVATE_METRIC_KEYS = frozenset({"hf_repo", "init_from_adapter_revision"})


def sanitize_worker_metrics(value: Any) -> Any:
    """Remove private artifact locators from worker metrics before persistence or reporting."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            name = str(key)
            if name in _PRIVATE_METRIC_KEYS and item:
                out[name] = _PRIVATE_SOURCE_MARKER
            elif name == "init_from_adapter" and isinstance(item, str) and item:
                try:
                    from flash.schema import parse_adapter_storage_ref

                    out[name] = (
                        _PRIVATE_SOURCE_MARKER
                        if parse_adapter_storage_ref(item) is not None
                        else item
                    )
                except Exception:
                    out[name] = _PRIVATE_SOURCE_MARKER if ":" in item else item
            else:
                out[name] = sanitize_worker_metrics(item)
        return out
    if isinstance(value, list):
        return [sanitize_worker_metrics(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_worker_metrics(item) for item in value]
    if isinstance(value, str):
        try:
            from flash.schema import parse_adapter_storage_ref

            if parse_adapter_storage_ref(value) is not None:
                return _PRIVATE_SOURCE_MARKER
        except Exception:
            pass
    return value


@dataclass
class RunMetrics:
    """Standard metrics record written per phase."""

    arm: str = "runpod"
    phase: str = ""  # "sft" | "rl" | "opd"
    # Completed optimizer updates (opd sets this; None for phases without a step count). Read by
    # _finalize to carry the true step onto the terminal `done` heartbeat so a cancel racing the DONE
    # upload doesn't re-price a fully-trained run to 0 steps.
    step: int | None = None
    seed: int = 0
    model_id: str = ""
    wall_seconds: float = 0.0  # training-loop wall time used for customer cost
    setup_seconds: float = 0.0  # cold start / provisioning + model load
    train_throughput_toks_per_s: float = 0.0
    train_tokens: int = 0
    generated_tokens: int = 0  # RL: total sampled completion tokens
    # cost_usd is stamped by runner._persist_metrics, not the worker.
    notes: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(sanitize_worker_metrics(asdict(self)), indent=2)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())
