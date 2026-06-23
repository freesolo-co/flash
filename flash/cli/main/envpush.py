"""Environment publish/install machinery for the `flash env` subcommands.

`flash env install` records a Freesolo environment id locally;
`flash env push` packages a local Freesolo environment and uploads it through the
managed Flash control plane.
"""

from __future__ import annotations

import sys
from pathlib import Path


def cmd_env_install(args) -> int:
    from flash.envs.adapter import is_github_environment_ref
    from flash.envs.registry import INSTALLED_MANIFEST, record_installed_env

    env_id = args.env_id
    if not is_github_environment_ref(env_id):
        print(
            "env id must be a Freesolo environment id / GitHub ref, e.g. "
            '"github:freesolo-co/environment-hub@main:user/project/publish-id/environment.py" '
            f"(got {env_id!r})",
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
_ENV_PUSH_SIDECAR_DIRS = frozenset({"assets", "data", "datasets", "db", "databases"})
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


def _push_env_name(raw: str) -> str:
    import re

    name = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return name or "flash-env"


def _env_name_from_ref_path(raw_path: str) -> str | None:
    path_text = raw_path.strip()
    if not path_text:
        return None
    path = Path(path_text)
    parts = path.as_posix().split("/")
    if path.name == _ENV_ENTRYPOINT and len(parts) >= 4:
        # Training-style refs are <namespace>/<project>/<publish-id>/environment.py.
        name = parts[-3]
    elif path.suffix == ".py":
        name = path.parent.name or path.stem
    else:
        name = path.name
    return name or None


def _config_env_name(config_path) -> str | None:
    """A stable name from a sibling flash.toml's `[environment] id`, or None."""
    import tomllib
    import urllib.parse

    path = Path(config_path)
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    env = data.get("environment")
    env_id = str(env.get("id") or "").strip() if isinstance(env, dict) else ""
    if env_id.startswith("github:"):
        _repo_ref, sep, ref_path = env_id[len("github:") :].partition(":")
        if not sep:
            return None
        return _env_name_from_ref_path(ref_path)

    parsed = urllib.parse.urlparse(env_id)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower() == "github.com":
        parts = [urllib.parse.unquote(part) for part in parsed.path.strip("/").split("/") if part]
        if len(parts) >= 5 and parts[2] in {"blob", "tree"}:
            return _env_name_from_ref_path("/".join(parts[4:]))
        return None

    if ":" in env_id and not parsed.scheme:
        return _env_name_from_ref_path(env_id.rsplit(":", 1)[1])
    return None


def _config_env_name_from_dir(config_dir) -> str | None:
    """The environment name declared by the sibling per-phase flash configs
    (``grpo.toml``/``sft.toml``). Without this, pushing ``environment.py`` finds no id
    and mints a brand-new env, so the run trains against the stale id in the configs.
    """
    config_dir = Path(config_dir)
    for cfg in ("grpo.toml", "sft.toml"):
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


_TAR_EXCLUDE_DIRS = _ENV_PUSH_IGNORED_NAMES


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


def _copy_env_sidecars(env_root: Path, dest: Path, *, entrypoint: Path) -> None:
    """Copy environment helpers and data sidecars, but not the agent source workspace."""
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
    import tempfile

    src = Path(args.path)
    if not src.exists():
        print(f"no such path: {src}", file=sys.stderr)
        return 1

    # A ready-made generated repo/package is reduced to the environment artifact. We intentionally
    # do not upload the agent-facing source workspace.
    if src.is_dir() and (src / "pyproject.toml").is_file():
        root_entrypoint = src / _ENV_ENTRYPOINT
        if root_entrypoint.is_file():
            env_root = src
            entrypoint = root_entrypoint
        else:
            print(
                f"{src} has a pyproject.toml but no environment.py entrypoint",
                file=sys.stderr,
            )
            return 1
        env_name = _pyproject_name(src) or _push_env_name(src.name)
        is_new = True

    # Wrap a bare Freesolo environment module (a single .py, or a one-module dir) into a compact
    # environment artifact and upload that. Sidecar datasets/DB files ship beside environment.py.
    elif src.is_file() and src.suffix == ".py":
        env_root = src.parent
        entrypoint = src
        # Re-publish to the same logical environment when sibling configs name one.
        sibling_name = _config_env_name_from_dir(src.parent)
        env_name = sibling_name or _push_env_name(src.stem)
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
        env_root = src
        entrypoint = modules[0]
        env_name = _push_env_name(src.name)
        is_new = True
    else:
        print(f"cannot publish {src}: expected a Freesolo .py module or an env directory.")
        return 1

    with tempfile.TemporaryDirectory(prefix="flash-env-push-") as tmp:
        pkg = Path(tmp)
        module_source = entrypoint.read_text()
        (pkg / _ENV_ENTRYPOINT).write_text(_with_syspath_bootstrap(module_source))
        _copy_env_sidecars(env_root, pkg, entrypoint=entrypoint)
        (pkg / "README.md").write_text(f"# {env_name}\n\nFlash Freesolo environment.\n")
        return _upload_and_report(env_name, is_new=is_new, package_b64=_tar_b64(pkg))
