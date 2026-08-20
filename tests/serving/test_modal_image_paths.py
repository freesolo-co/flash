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


def test_image_ships_the_package_under_its_real_import_path() -> None:
    """The container must be able to `import flash.serving.src.X`.

    Before the move these modules imported each other as `src.X`, so shipping the bare directory
    to `/root/src` was enough. They import each other as `flash.serving.src.X` now, and a `/root/src`
    tree cannot satisfy that: the container raises `ModuleNotFoundError: No module named
    'flash.serving'` on the first engine call -- *after* `modal deploy` has already reported success,
    which is why no offline test or deploy-time import catches it.
    """
    tree = _module()
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    # a bare directory mount cannot reconstruct the `flash.serving.src` package path.
    assert "add_local_dir" not in calls
    assert "add_local_python_source" in calls

    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_python_source"
    )
    assert [ast.literal_eval(arg) for arg in call.args] == ["flash"]

    # every module the engine imports must actually live under that package, or the mount ships a
    # tree that still cannot satisfy the imports.
    src = ROOT / "flash" / "serving" / "src"
    assert (ROOT / "flash" / "__init__.py").is_file()
    assert (ROOT / "flash" / "serving" / "__init__.py").is_file()
    assert (src / "__init__.py").is_file()

    # and modal_app's own module-scope flash imports must resolve inside that same package.
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("flash."):
            relative = Path(node.module.replace(".", "/"))
            assert (ROOT / f"{relative}.py").is_file() or (
                ROOT / relative / "__init__.py"
            ).is_file(), node.module
