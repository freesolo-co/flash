"""The hosted Modal app's on-disk paths must survive the move into flash.

`flash/serving/` used to be the top-level `serving/` of another repo, so every path it derives by
walking up from its own file shifted by one level, and its own `pyproject.toml` did not come with it.
Both breakages are invisible to the offline suite -- the app imports fine and every route test still
passes -- and only surface at `modal deploy`, which is the worst place to find them. These assert on
the source text because importing `modal_app` requires live Modal configuration.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODAL_APP = ROOT / "flash" / "serving" / "modal_app.py"


def _module() -> ast.Module:
    return ast.parse(MODAL_APP.read_text(encoding="utf-8"))


def _assigned_value(name: str) -> ast.expr:
    for node in ast.walk(_module()):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
    raise AssertionError(f"{name} is not assigned in modal_app.py")


def test_repo_dir_walks_up_to_the_actual_repo_root() -> None:
    # SERVING_DIR is flash/serving/, so the repo root is two parents up, not one. a single .parent
    # resolves to flash/, where load_dotenv finds no .env and silently returns -- a production deploy
    # then runs with none of its secrets and fails at request time rather than at deploy time.
    assert MODAL_APP.parent.parent.parent == ROOT
    assert (ROOT / "pyproject.toml").is_file()

    source = ast.unparse(_assigned_value("REPO_DIR"))
    assert source == "SERVING_DIR.parent.parent", source


def test_image_installs_from_a_pyproject_that_exists() -> None:
    # the app's own pyproject.toml did not survive the move; its dependency bounds live in flash's
    # `serve-runtime` and `serving` extras now. pip_install_from_pyproject on a missing file fails
    # the image build outright.
    call = next(
        node
        for node in ast.walk(_module())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pip_install_from_pyproject"
    )

    target = ast.unparse(call.args[0])
    assert "REPO_DIR" in target, target
    assert "pyproject.toml" in target, target
    assert "SERVING_DIR" not in target, target

    extras = next(kw for kw in call.keywords if kw.arg == "optional_dependencies")
    assert ast.literal_eval(extras.value) == ["serve-runtime", "serving"]

    # and those extras must genuinely exist, or the build resolves an empty dependency set.
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    for extra in ("serve-runtime", "serving"):
        assert declared[extra], extra
