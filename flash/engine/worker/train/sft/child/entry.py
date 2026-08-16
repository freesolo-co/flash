"""strict SFT child entrypoint for the verl external-module contract."""

from __future__ import annotations

import os
import sys


def main() -> None:
    import verl  # noqa: F401

    configured = os.environ.get("VERL_USE_EXTERNAL_MODULES")
    plugin = sys.modules.get("flash_sft_plugin")
    if configured != "flash_sft_plugin" or not getattr(plugin, "PLUGIN_LOADED_EXTERNALLY", False):
        raise RuntimeError(
            "flash_sft_plugin was not loaded by verl through VERL_USE_EXTERNAL_MODULES"
        )
    from verl.trainer.sft_trainer import main as sft_main

    sft_main()


if __name__ == "__main__":
    main()
