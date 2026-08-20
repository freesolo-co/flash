"""check the patched pinned verl agent-loop actor across ray serialization.

deliberately not a ci step: this imports ray and the pinned verl, neither of which is
installed on the cpu/offline ci runners, so wiring it into `ci.yml` would only ever fail.
run it by hand in a gpu worker environment (or any checkout with the verl extra) when
touching `install_exact_rollout_identity`:

    uv run python scripts/check_rollout_identity_actor.py

it catches the failure mode unit tests cannot: a patch that closes over something
non-picklable still passes in-process, then kills every grpo run at ray actor creation.
"""

from __future__ import annotations

import io
import sys
import types

import ray
from ray.util import inspect_serializability

from flash.engine.worker.train.rl.child import patches


def main() -> None:
    bridge = types.ModuleType("flash_grpo_multiturn")
    bridge.post_json = lambda *_args, **_kwargs: {}
    sys.modules["flash_grpo_multiturn"] = bridge

    from verl.experimental.agent_loop import agent_loop

    patches.install_exact_rollout_identity()
    actor = ray.remote(agent_loop.AgentLoopWorker)
    modified_class = actor.__ray_metadata__.modified_class
    payload = ray.cloudpickle.dumps(modified_class)
    diagnostics = io.StringIO()
    serializable, failures = inspect_serializability(
        modified_class,
        name="actual patched AgentLoopWorker actor class",
        depth=10,
        print_file=diagnostics,
    )
    if not serializable or failures:
        raise RuntimeError(diagnostics.getvalue())
    print(
        f"flash rollout identity actor serialization passed ({len(payload)} cloudpickle bytes)",
        flush=True,
    )


if __name__ == "__main__":
    main()
