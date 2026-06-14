"""Test helper: import a repo-root example's submodule by path.

Examples live at ``<repo>/examples/<name>/`` (outside the installed package and
not importable as ``autoslm.examples``), so tests load their modules the same
way ``autoslm.envs.registry`` does — as a uniquely-named package so relative
imports inside the example resolve.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"


def load_example_module(example: str, submodule: str | None = None):
    """Return ``examples/<example>/`` (or its ``<submodule>``) as a module."""
    pkg_name = f"autoslm_example_{example}"
    if pkg_name not in sys.modules:
        pkg_dir = _EXAMPLES_ROOT / example
        spec = importlib.util.spec_from_file_location(
            pkg_name, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = mod
        spec.loader.exec_module(mod)
    if submodule is None:
        return sys.modules[pkg_name]
    return importlib.import_module(f"{pkg_name}.{submodule}")
