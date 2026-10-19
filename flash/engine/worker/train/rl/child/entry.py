"""strict GRPO child entrypoint for the verl external-module contract."""

from __future__ import annotations

import os
import sys

if __name__ in {"__main__", "flash_grpo_entry"}:
    from flash_grpo_metrics import stream_metrics
else:
    from flash.engine.worker.train.rl.child.metrics import stream_metrics


def main() -> None:
    import verl  # noqa: F401

    configured = os.environ.get("VERL_USE_EXTERNAL_MODULES")
    plugin = sys.modules.get("flash_grpo_plugin")
    if configured != "flash_grpo_plugin" or not getattr(plugin, "PLUGIN_LOADED_EXTERNALLY", False):
        raise RuntimeError(
            "flash_grpo_plugin was not loaded by verl through VERL_USE_EXTERNAL_MODULES"
        )
    from verl.trainer.main_ppo import main as grpo_main

    with stream_metrics(os.environ["VERL_FILE_LOGGER_PATH"]):
        grpo_main()


if __name__ == "__main__":
    main()
