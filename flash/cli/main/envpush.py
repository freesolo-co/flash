"""Environment publish/install machinery for the `flash env` subcommands.

`flash env install` records a Freesolo environment ref locally;
`flash env push` packages a local Freesolo environment and uploads it through the
managed Flash control plane.
"""

from __future__ import annotations

import sys
from pathlib import Path


def cmd_env_install(args) -> int:
    from flash.envs.adapter import is_github_environment_ref
    from flash.envs.registry import record_installed_env

    env_id = args.env_id
    if not is_github_environment_ref(env_id):
        print(
            "env id must be a Freesolo environment ref, e.g. "
            '"github:owner/repo@main:path/to/freesolo/environment.py" '
            f"(got {env_id!r})",
            file=sys.stderr,
        )
        return 1
    record_installed_env(env_id, package="freesolo")
    print(f"installed {env_id}; recorded in ~/.flash/envs.json")
    print(f'use it via:  [environment]\\nid = "{env_id}"')
    return 0


# A Freesolo environment package is uploaded in canonical generated-repo layout:
# freesolo/environment.py exposes load_environment().
_ENV_PUSH_PYPROJECT = """\
[project]
name = "{name}"
version = "{version}"
description = "Flash Freesolo environment ({name})."
requires-python = ">=3.10"
dependencies = ["freesolo"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["freesolo"]
"""

_PUSH_INITIAL_VERSION = "0.1.0"


def _push_env_name(raw: str) -> str:
    import re

    name = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return name or "flash-env"


def _config_env_name(config_path) -> str | None:
    """A stable name from a sibling flash.toml's `[environment] id`, or None."""
    import tomllib

    path = Path(config_path)
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    env = data.get("environment")
    env_id = str(env.get("id") or "").strip() if isinstance(env, dict) else ""
    if ":" in env_id:
        if env_id.startswith("github:") and ":" not in env_id[len("github:") :]:
            return None
        path = Path(env_id.rsplit(":", 1)[1].strip())
        if path.name == "environment.py" and path.parent.name == "freesolo":
            name = path.parent.parent.name
        else:
            name = path.parent.name or path.stem
        return name or None
    return None


def _config_env_name_from_dir(config_dir) -> str | None:
    """The environment name declared by the sibling per-phase flash configs
    (``flash_grpo.toml``/``flash_sft.toml``). Without this, pushing ``environment.py`` finds
    no id and mints a brand-new env, so the run trains against the stale id in the configs.
    """
    config_dir = Path(config_dir)
    for cfg in ("flash_grpo.toml", "flash_sft.toml"):
        name = _config_env_name(config_dir / cfg)
        if name:
            return name
    return None


def _with_syspath_bootstrap(env_source: str) -> str:
    """Prepend a sys.path bootstrap so a published env (run as the package __init__) can resolve
    BARE absolute imports of its shipped sibling helpers (`import config` / `from utils import x`)
    even without its own sys.path.insert — otherwise load_environment fails
    with ModuleNotFoundError. Inserted AFTER the module docstring and any `from __future__` imports
    (which must stay first). Mirrors the platform hub publisher."""
    bootstrap = (
        "import os as _flash_os, sys as _flash_sys\n"
        "_flash_sys.path.insert(0, _flash_os.path.dirname(__file__))\n"
    )
    import ast

    try:
        tree = ast.parse(env_source)
    except SyntaxError:
        return bootstrap + env_source
    insert_after = 0
    body = tree.body
    i = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        insert_after = body[0].end_lineno or 0
        i = 1
    while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__":
        insert_after = body[i].end_lineno or insert_after
        i += 1
    lines = env_source.splitlines(keepends=True)
    return "".join(lines[:insert_after]) + bootstrap + "".join(lines[insert_after:])


_TAR_EXCLUDE_DIRS = frozenset({".prime", ".git", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache"})


def _tar_b64(directory: Path) -> str:
    """Pack a directory's contents into a base64 ``.tar.gz`` (members rooted at the top level)
    for upload, skipping tool/cache dirs (``.prime/``, ``.git/``, ``__pycache__``, ...). Packaging
    is pure file I/O and needs no local credentials; upload happens through the server.

    We walk with ``os.walk`` and PRUNE excluded directories in place (``dirs[:] = ...``) so we never
    descend into them: a plain ``rglob('*')`` would still recurse into (and stat every entry under)
    huge trees like ``.venv``/``.git`` only to discard them, making the push needlessly slow on a
    real project. The resulting member SET is identical to the previous filter on
    ``_TAR_EXCLUDE_DIRS`` (those dirs and everything beneath them are omitted), and the output is
    deterministic: entries are sorted at each level (parent-before-children walk order). Tar member
    order doesn't affect correctness — the server extracts in stream order and the content hash is
    computed from a re-sorted file list — so this is purely a traversal-efficiency fix."""
    import base64
    import io
    import os
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            # Prune excluded directories IN PLACE so os.walk never descends into them (this is the
            # whole fix — the excluded subtrees are never traversed/statted). Sort the survivors so
            # traversal order is deterministic.
            dirs[:] = sorted(d for d in dirs if d not in _TAR_EXCLUDE_DIRS)
            # Add the surviving directory entries themselves (so empty dirs are preserved, as the old
            # rglob walk did), then this level's files — both sorted for determinism. recursive=False:
            # we walk every path ourselves, so letting tar recurse would add contents twice.
            for name in dirs:
                child = root_path / name
                tar.add(child, arcname=str(child.relative_to(directory)), recursive=False)
            for name in sorted(files):
                child = root_path / name
                tar.add(child, arcname=str(child.relative_to(directory)), recursive=False)
    return base64.b64encode(buf.getvalue()).decode()


def _pyproject_name(env_dir: Path) -> str | None:
    """The ``[project] name`` of a ready-made env dir, used to name the managed env."""
    import tomllib

    try:
        data = tomllib.loads((env_dir / "pyproject.toml").read_text())
        name = (data.get("project") or {}).get("name")
        return str(name) if name else None
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return None


def _upload_and_report(name: str, *, is_new: bool, package_b64: str) -> int:
    """Upload a packaged env to the managed control plane and print the returned id."""
    from flash.client import ClientError, client_from_config

    try:
        result = client_from_config().publish_env(name=name, is_new=is_new, package_b64=package_b64)
    except ClientError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    slug = result.get("id")
    if not slug:
        print("warning: the env was uploaded but the server returned no id", file=sys.stderr)
        return 1
    print(f"published {slug}")
    print(f'reference it in your config:\n\n  [environment]\n  id = "{slug}"')
    return 0


def cmd_env_push(args) -> int:
    import shutil
    import tempfile

    src = Path(args.path)
    if not src.exists():
        print(f"no such path: {src}", file=sys.stderr)
        return 1

    # A ready-made generated repo/package is uploaded as-is when it already carries the canonical
    # Freesolo environment file.
    if src.is_dir() and (src / "pyproject.toml").is_file():
        if not (src / "freesolo" / "environment.py").is_file():
            print(
                f"{src} has a pyproject.toml but no freesolo/environment.py entrypoint",
                file=sys.stderr,
            )
            return 1
        env_name = _pyproject_name(src) or _push_env_name(src.name)
        return _upload_and_report(env_name, is_new=True, package_b64=_tar_b64(src))

    # Wrap a bare Freesolo environment module (a single .py, or a one-module dir) into the
    # canonical generated-repo layout and upload that. `data_dir` is a committed `datasets/`
    # sibling of the module; it ships beside the environment.
    if src.is_file() and src.suffix == ".py":
        module_source = src.read_text()
        # Re-publish to the same logical environment when sibling configs name one.
        sibling_name = _config_env_name_from_dir(src.parent)
        env_name = sibling_name or _push_env_name(src.stem)
        data_dir = src.parent / "datasets"
        # Ship the env's sibling helper modules (config.py/utils.py/...) so an environment.py that
        # does `sys.path.insert(0, dir(__file__)); import utils` resolves once installed.
        sibling_modules = [
            p for p in sorted(src.parent.glob("*.py")) if p != src and not p.name.startswith("__")
        ]
        is_new = sibling_name is None
    elif src.is_dir():
        modules = [p for p in sorted(src.glob("*.py")) if not p.name.startswith("__")]
        if len(modules) != 1:
            print(
                f"{src} has no pyproject.toml and {'no' if not modules else 'multiple'} "
                "top-level .py module(s); point `flash env push` at the env's .py file or add a "
                "pyproject.toml.",
                file=sys.stderr,
            )
            return 1
        module_source = modules[0].read_text()
        env_name = _push_env_name(src.name)
        data_dir = src / "datasets"
        sibling_modules = []
        is_new = True
    else:
        print(f"cannot publish {src}: expected a Freesolo .py module or an env directory.")
        return 1

    with tempfile.TemporaryDirectory(prefix="flash-env-push-") as tmp:
        pkg = Path(tmp)
        env_pkg = pkg / "freesolo"
        env_pkg.mkdir()
        (env_pkg / "__init__.py").write_text("")
        (env_pkg / "environment.py").write_text(_with_syspath_bootstrap(module_source))
        # Ship committed sibling data inside the canonical freesolo package dir.
        if data_dir.is_dir() and any(data_dir.iterdir()):
            shutil.copytree(data_dir, env_pkg / "datasets")
        for mod in sibling_modules:
            shutil.copy2(mod, env_pkg / mod.name)
        (pkg / "pyproject.toml").write_text(
            _ENV_PUSH_PYPROJECT.format(name=env_name, version=_PUSH_INITIAL_VERSION)
        )
        (pkg / "README.md").write_text(f"# {env_name}\n\nFlash Freesolo environment.\n")
        return _upload_and_report(env_name, is_new=is_new, package_b64=_tar_b64(pkg))
