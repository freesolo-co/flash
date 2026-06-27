"""Cross-package contract: autoenv depends on stable flash surfaces, one-directionally.

If a flash refactor renames/removes a surface autoenv reuses, this fails in CI before the
harness breaks. And flash must never import autoenv (preserving the zero-dep client).
"""

from __future__ import annotations

import re
from pathlib import Path

import flash

# An actual import of the autoenv package — `import autoenv...` or `from autoenv... import ...`.
# Deliberately matches import statements, not the bare substring "autoenv" (which could appear
# in a comment or help string without creating a real dependency).
_IMPORTS_AUTOENV = re.compile(r"^\s*(?:import\s+autoenv|from\s+autoenv)\b", re.MULTILINE)


def test_flash_surfaces_autoenv_depends_on_exist():
    import flash.catalog as catalog
    import flash.cli.commands as commands
    import flash.cli.training_doc as training_doc
    import flash.client.http as http
    import flash.cost.spec as cost_spec
    import flash.engine.vram as vram
    import flash.schema as schema

    assert hasattr(catalog, "MODELS")
    assert hasattr(catalog, "ALGORITHMS")
    assert hasattr(catalog, "list_models")
    assert hasattr(catalog, "validate_model_for_algorithm")
    assert hasattr(cost_spec, "estimate_for_spec")
    assert hasattr(vram, "check_fit")
    assert hasattr(vram, "resolve_params_b")
    assert hasattr(schema, "spec_from_file")
    assert hasattr(schema, "spec_from_dict")
    assert hasattr(commands, "cmd_env_setup")
    assert isinstance(training_doc.TRAINING_MD, str)
    assert "TRAINING.md" in training_doc.TRAINING_MD

    for method in ("publish_env", "create_run", "get_run", "deploy", "chat"):
        assert hasattr(http.ApiClient, method), method


def test_flash_never_imports_autoenv():
    flash_root = Path(flash.__file__).parent
    offenders = [
        str(py.relative_to(flash_root))
        for py in flash_root.rglob("*.py")
        if _IMPORTS_AUTOENV.search(py.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"flash modules must not import autoenv: {offenders}"
