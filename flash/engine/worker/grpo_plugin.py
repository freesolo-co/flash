"""child-side entrypoint that registers flash's multi-turn GRPO agent loop with verl.

runs INSIDE the verl interpreter, loaded through ``VERL_USE_EXTERNAL_MODULES`` (verl/__init__.py
calls ``import_external_libs`` on it at import time, before the trainer builds its rollout). this
module is the only place a verl symbol is imported, which is why the loop itself lives in a
separate stdlib-only module behind a ``build_*`` factory: the parent process can import, lint, and
unit-test that module without verl installed.
"""

from __future__ import annotations

try:  # inside the verl child, copied in beside this file
    from flash_grpo_multiturn import build_flash_grpo_multi_turn_agent_loop
except ImportError:  # in-tree (parent process, tests, lint)
    from flash.engine.worker.grpo_multiturn import build_flash_grpo_multi_turn_agent_loop


def install() -> None:
    """register the flash multi-turn agent loop under the name the hydra override selects."""
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopOutput,
        register,
    )

    build_flash_grpo_multi_turn_agent_loop(
        register=register,
        agent_loop_base=AgentLoopBase,
        agent_loop_output=AgentLoopOutput,
    )


install()
