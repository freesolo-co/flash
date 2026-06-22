"""Environment publish/install machinery for the `flash env` subcommands."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Published environment wheel index. Each org's wheels live under ITS OWN namespace
# (e.g. freesolo-co/flash-bench -> .../freesolo-co/simple/), so derive the index from the
# slug owner; a single hardcoded owner index 404s for other orgs.
PRIME_HUB_INDEX_TMPL = "https://hub.primeintellect.ai/{owner}/simple/"
_INSTALL_ERROR_LIMIT = 4000


def _prime_hub_index(env_id: str) -> str:
    owner = env_id.split("/", 1)[0] if "/" in env_id else "primeintellect"
    return PRIME_HUB_INDEX_TMPL.format(owner=owner)


def _trim_install_output(stdout: str | None, stderr: str | None) -> str:
    detail = "\n".join(part.strip() for part in (stderr, stdout) if part and part.strip())
    if len(detail) > _INSTALL_ERROR_LIMIT:
        return f"...\n{detail[-_INSTALL_ERROR_LIMIT:]}"
    return detail


def cmd_env_install(args) -> int:
    import shutil
    import subprocess

    from flash.envs.registry import _bare_wheel_name, record_installed_env

    env_id = args.env_id
    # Managed envs are published environment ids: exactly one `/` with non-empty owner and name.
    # A bare id (`gsm8k`) or a malformed id can't be resolved, so reject it up front
    # rather than letting the installer fail with an opaque error.
    parts = env_id.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        print(
            f'env id must be "owner/name" (got {env_id!r})',
            file=sys.stderr,
        )
        return 1
    # `flash env install` is a LOCAL-client convenience: it installs the env into the client's
    # interpreter and records it in ~/.flash/envs.json for local authoring/dry-run. The
    # managed worker does NOT reinstall from this record; it installs published envs itself on the
    # GPU box. A published id `owner/name` maps to the pip wheel `name`; we record the index
    # alongside the env.
    extras = {"extra_index_url": _prime_hub_index(env_id)}
    if shutil.which("prime"):
        # The private installer resolves the environment + index itself, and is the only path that
        # can fetch private env wheels.
        cmd = ["prime", "env", "install", env_id]
    else:
        print(f"installing {env_id} locally")
        installer = (
            # `uv pip install` outside an active venv errors with "No virtual environment
            # found"; --python targets the CLI's own interpreter so a global/pipx `flash`
            # install still records the env.
            ["uv", "pip", "install", "--python", sys.executable]
            if shutil.which("uv")
            else [sys.executable, "-m", "pip", "install"]
        )
        cmd = [*installer, _bare_wheel_name(env_id), "--extra-index-url", extras["extra_index_url"]]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rc = proc.returncode
    if rc != 0:
        print("install failed", file=sys.stderr)
        detail = _trim_install_output(getattr(proc, "stdout", None), getattr(proc, "stderr", None))
        if detail:
            print(detail, file=sys.stderr)
        return rc
    record_installed_env(env_id, package=_bare_wheel_name(env_id), extras=extras)
    print(f"installed {env_id}; recorded in ~/.flash/envs.json")
    print(f'use it via:  [environment]\\nid = "{env_id}"')
    return 0


# A packaged verifiers env is a pyproject + an importable module exposing
# load_environment(). When `flash env push` is pointed at a bare module (a single `.py`, as the
# freesolo training agent emits, or a dir without a pyproject), we wrap it in this layout so the
# push Just Works instead of erroring on "pyproject.toml not found".
_ENV_PUSH_PYPROJECT = """\
[project]
name = "{name}"
version = "{version}"
description = "Flash verifiers environment ({name})."
requires-python = ">=3.10"
dependencies = ["verifiers"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["{module}"]
"""

_PUSH_INITIAL_VERSION = "0.1.0"


def _push_env_name(raw: str) -> str:
    import re

    name = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return name or "flash-env"


def _config_env_name(config_path) -> str | None:
    """The `name` part of a sibling flash.toml's `[environment] id = "owner/name"`, or None.

    Used so a bare `environment.py` re-publishes under its EXISTING env (minting a new
    version) instead of deriving a fresh name from the file stem. Owner still comes from the
    authenticated managed account/team, so only the name part is consumed here."""
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
    if "/" in env_id:
        name = env_id.split("/", 1)[1].strip()
        return name or None
    return None


def _config_env_name_from_dir(config_dir) -> str | None:
    """The published env name declared by the sibling per-phase flash configs
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
    even without its own sys.path.insert — otherwise install/load_environment fails
    with ModuleNotFoundError. Inserted AFTER the module docstring and any `from __future__` imports
    (which must stay first). Mirrors the platform hub publisher."""
    bootstrap = (
        "import os as _flash_os, sys as _flash_sys\n"
        "_flash_sys.path.insert(0, _flash_os.path.dirname(__file__))\n"
    )
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


# Tool/cache dirs that aren't part of the environment SOURCE. We never ship them: `.prime/` in
# particular carries installer metadata (.env-metadata.json) from a prior local push — shipping it
# bloats the upload and could let stale client metadata confuse server-side slug discovery (the
# server also strips it defensively, but don't send it in the first place).
_TAR_EXCLUDE_DIRS = frozenset({".prime", ".git", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache"})


def _tar_b64(directory: Path) -> str:
    """Pack a directory's contents into a base64 ``.tar.gz`` (members rooted at the top level)
    for upload, skipping tool/cache dirs (``.prime/``, ``.git/``, ``__pycache__``, ...). Packaging
    is pure file I/O; no external env CLI or account is needed locally.

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
    """Upload a packaged env to the managed service and print the returned id."""
    from flash.client import ClientError, client_from_config

    try:
        result = client_from_config().publish_env(name=name, is_new=is_new, package_b64=package_b64)
    except ClientError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    slug = result.get("id")
    if not slug:
        print("warning: the env was published but the server returned no id", file=sys.stderr)
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

    # A ready-made env directory (has a pyproject.toml) is uploaded as-is; its name comes from
    # the pyproject. Publishing happens server-side under FreeSolo's managed account, so no
    # separate environment account is required.
    if src.is_dir() and (src / "pyproject.toml").is_file():
        env_name = _pyproject_name(src) or _push_env_name(src.name)
        return _upload_and_report(env_name, is_new=True, package_b64=_tar_b64(src))

    # Wrap a bare verifiers module (a single .py, or a one-module dir) into a compatible
    # env package and upload that. `data_dir` is a committed `datasets/` sibling of the module
    # (if any); we ship it inside the package so an env that reads a `__file__`-relative data
    # file still resolves once installed.
    if src.is_file() and src.suffix == ".py":
        module_source = src.read_text()
        # Re-publish to the SAME env when a sibling flash config names one: use its
        # `[environment] id` name part so an edited environment.py mints a new version of the
        # existing env instead of creating a fresh env from the file stem.
        sibling_name = _config_env_name_from_dir(src.parent)
        env_name = sibling_name or _push_env_name(src.stem)
        data_dir = src.parent / "datasets"
        # Ship the env's sibling helper modules (config.py/utils.py/...) so an environment.py that
        # does `sys.path.insert(0, dir(__file__)); import utils` resolves once installed.
        sibling_modules = [
            p for p in sorted(src.parent.glob("*.py")) if p != src and not p.name.startswith("__")
        ]
        # A sibling config id means we're re-publishing an EXISTING env: auto-bump from the
        # first attempt so it doesn't restart at 0.1.0 and climb through version conflicts.
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
        print(f"cannot publish {src}: expected a verifiers .py module or an env directory.")
        return 1

    # The module dir name must be a valid Python identifier. `env_name` may be a sibling config's
    # published id name (`_config_env_name`), which is NOT sanitized and can contain `.`/other chars
    # invalid in a package dir (and would mismatch `[tool.hatch...] packages = ["<module>"]`,
    # breaking the build). Normalize through `_push_env_name` (collapses non-[a-z0-9] runs to `-`)
    # before mapping `-`->`_`, so the module is always [a-z0-9_].
    module = _push_env_name(env_name).replace("-", "_")
    # A Python package name can't start with a digit, so prefix one (e.g. "2026-task").
    if module[:1].isdigit():
        module = f"env_{module}"
    with tempfile.TemporaryDirectory(prefix="flash-env-push-") as tmp:
        pkg = Path(tmp)
        (pkg / module).mkdir()
        (pkg / module / "__init__.py").write_text(_with_syspath_bootstrap(module_source))
        # Ship committed sibling data inside the package dir (it lands at <module>/datasets/, so a
        # `os.path.dirname(__file__)/datasets/...` read resolves on the worker); the whole package
        # dir ships via `[tool.hatch.build.targets.wheel] packages = ["<module>"]`.
        if data_dir.is_dir() and any(data_dir.iterdir()):
            shutil.copytree(data_dir, pkg / module / "datasets")
        for mod in sibling_modules:
            shutil.copy2(mod, pkg / module / mod.name)
        (pkg / "pyproject.toml").write_text(
            _ENV_PUSH_PYPROJECT.format(name=env_name, module=module, version=_PUSH_INITIAL_VERSION)
        )
        (pkg / "README.md").write_text(f"# {env_name}\n\nFlash verifiers environment.\n")
        return _upload_and_report(env_name, is_new=is_new, package_b64=_tar_b64(pkg))
