"""strict OPD child entrypoint for the verl external-module contract."""

from __future__ import annotations

import os
import sys


def main() -> None:
    import verl  # noqa: F401

    configured = os.environ.get("VERL_USE_EXTERNAL_MODULES")
    plugin = sys.modules.get("flash_opd_plugin")
    if configured != "flash_opd_plugin" or not getattr(plugin, "PLUGIN_LOADED_EXTERNALLY", False):
        raise RuntimeError(
            "flash_opd_plugin was not loaded by verl through VERL_USE_EXTERNAL_MODULES"
        )
    plugin.main()


if __name__ == "__main__":
    main()
