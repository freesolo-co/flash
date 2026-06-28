"""Cost accounting and run-metrics record for Flash runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class RunMetrics:
    """Standard metrics record written per phase."""

    arm: str = "runpod"
    phase: str = ""  # "sft" | "rl"
    seed: int = 0
    model_id: str = ""
    wall_seconds: float = 0.0
    setup_seconds: float = 0.0  # cold start / provisioning + model load
    train_throughput_toks_per_s: float = 0.0
    train_tokens: int = 0
    generated_tokens: int = 0  # RL: total sampled completion tokens
    # cost_usd is stamped by runner._persist_metrics, not the worker.
    notes: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())
