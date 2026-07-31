"""GRPO entry point.

``run_rl`` delegates to ``rl_verl.run_rl_verl``, which owns config, dataset prep, rollout, reward
bridging, and training. verl is the only GRPO backend, so nothing else lives here.
"""

from __future__ import annotations


def run_rl():
    """Run GRPO on verl."""
    from flash.engine.worker.rl_verl import run_rl_verl

    run_rl_verl()
