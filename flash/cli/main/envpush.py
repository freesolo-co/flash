"""Environment publish/install machinery for the `flash env` subcommands.

`flash env install` records a Freesolo environment id locally;
`flash env push` packages a local Freesolo environment and uploads it through the
managed Flash control plane.
"""

from __future__ import annotations

import sys
from pathlib import Path


def cmd_env_install(args) -> int:
    from flash.envs.adapter import is_freesolo_environment_id
    from flash.envs.registry import INSTALLED_MANIFEST, record_installed_env

    env_id = args.env_id
    if not is_freesolo_environment_id(env_id):
        print(
            f'env id must be a Freesolo environment id, e.g. "your-name/your-env" (got {env_id!r})',
            file=sys.stderr,
        )
        return 1
    record_installed_env(env_id, package="freesolo")
    print(f"installed {env_id}; recorded in {INSTALLED_MANIFEST}")
    print(f'use it via:  [environment]\\nid = "{env_id}"')
    return 0


_ENV_ENTRYPOINT = "environment.py"
_ENV_PUSH_IGNORED_NAMES = frozenset(
    {
        ".prime",
        ".git",
        ".github",
        "__pycache__",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        "source",
    }
)
_ENV_PUSH_SIDECAR_DIRS = frozenset({"assets", "data", "databases", "datasets", "db"})
_ENV_PUSH_SIDECAR_SUFFIXES = frozenset(
    {
        ".csv",
        ".db",
        ".json",
        ".jsonl",
        ".parquet",
        ".pkl",
        ".sqlite",
        ".sqlite3",
        ".sql",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)


def _normalize_env_name(raw: str) -> str | None:
    import re

    name = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return name or None


def _with_syspath_bootstrap(env_source: str) -> str:
    """Prepend a sys.path bootstrap so a published env can resolve shipped sibling helpers."""
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


_TAR_EXCLUDE_DIRS = _ENV_PUSH_IGNORED_NAMES


def _tar_b64(directory: Path) -> str:
    """Pack a directory into a base64 tarball, excluding caches and metadata directories."""
    import base64
    import io
    import os
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            dirs[:] = sorted(d for d in dirs if d not in _TAR_EXCLUDE_DIRS)
            for name in dirs:
                child = root_path / name
                tar.add(child, arcname=str(child.relative_to(directory)), recursive=False)
            for name in sorted(files):
                child = root_path / name
                tar.add(child, arcname=str(child.relative_to(directory)), recursive=False)
    return base64.b64encode(buf.getvalue()).decode()


def _copy_env_sidecars(env_root: Path, dest: Path, *, entrypoint: Path) -> None:
    """Copy helper code and data sidecars beside environment.py."""
    import shutil

    for child in sorted(env_root.iterdir()):
        if child == entrypoint or child.name in _ENV_PUSH_IGNORED_NAMES:
            continue
        if child.name.startswith("."):
            continue
        target = dest / child.name
        if child.is_dir():
            if child.name in _ENV_PUSH_SIDECAR_DIRS:
                shutil.copytree(
                    child,
                    target,
                    ignore=shutil.ignore_patterns(*_ENV_PUSH_IGNORED_NAMES),
                )
            continue
        if not child.is_file():
            continue
        if (
            child.suffix == ".py" and not child.name.startswith("__")
        ) or child.suffix.lower() in _ENV_PUSH_SIDECAR_SUFFIXES:
            shutil.copy2(child, target)


def _upload_and_report(name: str, *, package_b64: str) -> int:
    """Upload a packaged env to the managed control plane and print the returned id."""
    from flash.client import ClientError, client_from_config

    try:
        result = client_from_config().publish_env(name=name, package_b64=package_b64)
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
    import tempfile

    env_name = _normalize_env_name(str(getattr(args, "name", "") or ""))
    if not env_name:
        print("env name required: pass `--name <name>`", file=sys.stderr)
        return 1

    src = Path(args.path)
    if not src.exists():
        print(f"no such path: {src}", file=sys.stderr)
        return 1

    if src.is_dir():
        canonical_entrypoint = src / _ENV_ENTRYPOINT
        if canonical_entrypoint.is_file():
            entrypoint = canonical_entrypoint
            env_root = src
        elif (src / "pyproject.toml").is_file():
            print(f"{src} has a pyproject.toml but no environment.py entrypoint", file=sys.stderr)
            return 1
        else:
            modules = [p for p in sorted(src.glob("*.py")) if not p.name.startswith("__")]
            if len(modules) != 1:
                print(
                    f"{src} has no environment.py and "
                    f"{'no' if not modules else 'multiple'} top-level .py module(s); "
                    "add an environment.py entrypoint or pass the exact .py file "
                    "for a single-file smoke test.",
                    file=sys.stderr,
                )
                return 1
            env_root = src
            entrypoint = modules[0]
    elif src.is_file() and src.suffix == ".py":
        env_root = src.parent
        entrypoint = src
    else:
        print(
            f"cannot publish {src}: expected a Freesolo .py module or an env directory.",
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory(prefix="flash-env-push-") as tmp:
        pkg = Path(tmp)
        module_source = entrypoint.read_text()
        (pkg / _ENV_ENTRYPOINT).write_text(_with_syspath_bootstrap(module_source))
        _copy_env_sidecars(env_root, pkg, entrypoint=entrypoint)
        (pkg / "README.md").write_text(f"# {env_name}\n\nFlash Freesolo environment.\n")
        return _upload_and_report(env_name, package_b64=_tar_b64(pkg))
