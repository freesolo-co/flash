"""Cost accounting + the standard run-metrics record for AutoSLM runs.

GPU cost = gpu_hours * hourly_rate (RunPod per-second billing; artifacts go via HF).
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
    prefill_tokens: int = 0
    sample_tokens: int = 0
    train_tokens: int = 0
    generated_tokens: int = 0  # RL: total sampled completion tokens
    # Cost
    cost_usd: float = 0.0
    gpu_seconds: float = 0.0  # GPU-rental wall seconds
    # Quality
    base_eval_acc: float | None = None  # baseline (no adapter) on this substrate
    trained_eval_acc: float | None = None
    # Misc / friction
    notes: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())
