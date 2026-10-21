"""GRPO entry point.

``run_rl`` delegates to ``rl_train.run_rl_train``, which owns config, dataset prep, rollout, reward
bridging, and training. verl is the only GRPO backend, so nothing else lives here.
"""

from __future__ import annotations


def run_rl():
    """Run GRPO on verl."""
    from flash.engine.worker.train.entry.rl_train import run_rl_train

    run_rl_train()
