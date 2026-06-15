"""Cost accounting + the standard run-metrics record for AutoSLM runs.

GPU cost = gpu_hours * hourly_rate (per-second billing on the selected provider —
RunPod or Vast.ai; artifacts go via HF).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class RunMetrics:
    """Standard metrics record written per phase/seed."""

    arm: str = "runpod"  # compute substrate
    phase: str = ""  # "sft" | "rl"
    seed: int = 0
    model_id: str = ""
    # Speed
    wall_seconds: float = 0.0
    setup_seconds: float = 0.0  # cold start / provisioning + model load
    train_throughput_toks_per_s: float = 0.0
    # Token accounting
    train_tokens: int = 0
    generated_tokens: int = 0  # RL: total sampled completion tokens
    # Misc / friction. cost_usd is computed/stamped downstream by the runner from the
    # provider's $/hr (see runner._persist_metrics), not by the worker.
    notes: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())
