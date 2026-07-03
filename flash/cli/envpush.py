"""Environment publish/pull machinery for the `flash env` subcommands.

`flash env push` packages a local Freesolo environment and uploads it through the
managed Flash control plane.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from . import render
from ._tty import TtyStatusLine

if TYPE_CHECKING:
    from flash.client.http import ProgressCallback


def _atomic_write_bytes(out: Path, data: bytes | bytearray) -> None:
    """Write data via a sibling temp file so existing files are not truncated on failure."""
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".flash-env-pull-")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, out)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _err(msg: str) -> int:
    """Print a command error (themed red ✗ on a TTY, plain on the machine path) and return 1.

    These `env` failures `return 1` directly rather than raising, so they never reach main()'s
    themed handler; this keeps them on the same red ✗ idiom while leaving the machine path's exact
    text untouched for scripts and the env tests."""
    print(render.error(msg) if render.styled() else msg, file=sys.stderr)
    return 1


_MANAGED_HUB_ID_PART_RE = re.compile(r"[a-z0-9._-]+")
_ManagedHubIdError: TypeAlias = Literal["not-managed", "not-canonical"]


def _normalize_managed_hub_id(raw: object) -> tuple[str | None, _ManagedHubIdError | None]:
    from flash.envs.adapter import is_managed_environment_slug

    env_id = str(raw or "").strip()
    if not is_managed_environment_slug(env_id):
        return None, "not-managed"
    if env_id != env_id.lower() or not all(
        _MANAGED_HUB_ID_PART_RE.fullmatch(part) for part in env_id.split("/")
    ):
        return None, "not-canonical"
    return env_id, None


def cmd_env_pull(args) -> int:
    """Download a published Freesolo environment (or a single file from it) to local disk.

    Environments are addressed by their managed hub slug ``namespace/name`` and pulled as package
    tarballs through the authenticated Flash control plane.
    """
    from flash.client import ClientError, client_from_config
    from flash.envs.pull import (
        download_environment_file_from_archive,
        ensure_environment_pull_destination_available,
        environment_local_dirname,
        pull_environment_package_from_archive,
    )

    env_id = str(args.env_id or "").strip()
    managed_env_id, managed_error = _normalize_managed_hub_id(env_id)
    if managed_error == "not-canonical":
        return _err(
            f'env id must be lowercase "namespace/name" with no spaces (got {args.env_id!r})'
        )
    if managed_env_id is None:
        print(
            'env id must be a managed Freesolo hub slug "your-name/your-env" '
            f"(got {args.env_id!r})",
            file=sys.stderr,
        )
        return 1
    env_id = managed_env_id
    try:
        if args.path:
            default_name = Path(args.path.replace("\\", "/")).name
            out = Path(args.output) if args.output else Path(default_name)
            if out.is_dir() and not out.is_symlink():
                print(
                    f"refusing to overwrite directory {out} with a file "
                    "(a single-file pull needs -o to be a FILE path, not a directory)",
                    file=sys.stderr,
                )
                return 1
            if (out.exists() or out.is_symlink()) and not args.force:
                print(f"refusing to overwrite {out} (pass --force)", file=sys.stderr)
                return 1
            package = client_from_config().download_env_package(env_id)
            data = download_environment_file_from_archive(package, args.path)
            out.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(out, data)
            if render.styled():
                print(render.env_pulled(str(out), f"{args.path} · {len(data):,} bytes"))
            else:
                print(f"pulled {args.path} from {env_id} -> {out} ({len(data):,} bytes)")
        else:
            out = Path(args.output) if args.output else Path(environment_local_dirname(env_id))
            ensure_environment_pull_destination_available(out, overwrite=args.force)
            package = client_from_config().download_env_package(env_id)
            pull_environment_package_from_archive(package, out, overwrite=args.force)
            if render.styled():
                print(render.env_pulled(f"{out}/", env_id))
            else:
                print(f"pulled {env_id} -> {out}/")
        return 0
    except FileExistsError:
        print(
            f"refusing to overwrite existing {args.output or environment_local_dirname(env_id)!r} "
            "(pass --force)",
            file=sys.stderr,
        )
        return 1
    except ClientError as exc:
        print(f"env pull failed: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"env pull failed: {exc}", file=sys.stderr)
        return 1


def cmd_env_delete(args) -> int:
    from flash.client import ClientError, client_from_config

    # Delete only targets MANAGED hub ids ("namespace/name") — not github: refs or local paths, which
    # don't live on the hub. And enforce the hub's canonical form (lowercase, no surrounding
    # whitespace) BEFORE the network call: the server's slug validator is lowercase-only, so a
    # mixed-case / padded id would otherwise make a pointless request and return a confusing 400.
    env_id, validation_error = _normalize_managed_hub_id(args.env_id)
    if validation_error == "not-managed":
        return _err(
            f'env id must be a managed Freesolo hub id "namespace/name" (got {args.env_id!r}); '
            "github refs and local paths can't be deleted from the hub"
        )
    if validation_error == "not-canonical" or env_id is None:
        return _err(
            f'env id must be lowercase "namespace/name" with no spaces (got {args.env_id!r})'
        )
    if not getattr(args, "yes", False):
        prompt = f"delete environment {env_id}? this removes it from the hub [y/N] "
        try:
            answer = input(render.warn(prompt) if render.styled() else prompt)
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("aborted; environment not deleted", file=sys.stderr)
            return 1
    try:
        result = client_from_config().delete_env(env_id)
    except ClientError as exc:
        return _err(str(exc))
    deleted = bool(result["deleted"])
    msg = f"deleted {env_id}" if deleted else f"{env_id} was not found on the hub (already deleted)"
    print(render.ok(msg) if render.styled() else msg)
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
        "pyproject.toml",
        "source",
    }
)
# ``dataset`` is what ``flash env setup`` scaffolds; ``datasets`` (plural) is the more common
# convention users reach for by hand. Accept both so a data dir under either name ships in the
# tarball — a name outside this allowlist is silently dropped, which would train the worker on
# missing rows.
_ENV_PUSH_SIDECAR_DIRS = frozenset({"dataset", "datasets"})
# ``.md`` is included so the ``TRAINING.md`` playbook `flash env setup` scaffolds (and any
# user-authored README/NOTES) travels with the env into the hub and back out through
# ``flash env pull`` — a published env should carry its own training guidance, not just code+data.
_ENV_PUSH_SIDECAR_SUFFIXES = frozenset(
    {
        ".csv",
        ".json",
        ".jsonl",
        ".md",
        ".parquet",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)


def _normalize_env_name(raw: str) -> str | None:
    """Normalize only the NAME segment; a namespace prefix is passed through verbatim.

    The server is the sole authority on namespace grammar and ownership — rewriting the
    namespace client-side would silently target a different (possibly forbidden) slug.
    """
    from flash.schema import normalize_env_name_segment

    text = str(raw or "").strip()
    if "/" not in text:
        return normalize_env_name_segment(text)
    parts = [part.strip() for part in text.split("/")]
    if len(parts) != 2 or not all(parts):
        return None
    name = normalize_env_name_segment(parts[1])
    if name is None:
        return None
    return f"{parts[0]}/{name}"


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
        if (
            child == entrypoint
            or child.name == _ENV_ENTRYPOINT
            or child.name in _ENV_PUSH_IGNORED_NAMES
        ):
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


def _human_bytes(n: int) -> str:
    """Human-readable byte count."""
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class _UploadProgress(TtyStatusLine):
    """Carriage-return upload progress bar on stderr; no-op off a TTY."""

    _BAR_WIDTH = 24

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._last_pct = -1

    @property
    def callback(self) -> ProgressCallback | None:
        return self.update if self._enabled else None

    def status(self, message: str) -> None:
        if self._enabled:
            self._write(message)

    def update(self, sent: int, total: int) -> None:
        if not self._enabled:
            return
        pct = 100 if total <= 0 else min(100, sent * 100 // total)
        # avoid thousands of redraws: skip if percent unchanged mid-upload
        if pct == self._last_pct and sent < total:
            return
        self._last_pct = pct
        filled = (
            self._BAR_WIDTH if total <= 0 else min(self._BAR_WIDTH, self._BAR_WIDTH * sent // total)
        )
        bar = "#" * filled + "-" * (self._BAR_WIDTH - filled)
        self._write(
            f"uploading {self._name} [{bar}] {pct:3d}% {_human_bytes(sent)}/{_human_bytes(total)}"
        )


def _upload_and_report(name: str, *, package_b64: str, bar: _UploadProgress | None = None) -> int:
    """Upload a packaged env to the managed control plane and print the returned id."""
    from flash.client import ClientError, client_from_config

    bar = bar or _UploadProgress(name)
    try:
        result = client_from_config().publish_env(
            name=name, package_b64=package_b64, progress=bar.callback
        )
    except ClientError as exc:
        bar.clear()
        return _err(str(exc))
    bar.clear()
    slug = result.get("id")
    if not slug:
        # a publish that can't report an id has not done its job (no `id = "..."` snippet to show),
        # so this is a hard failure — route it through _err's red ✗ idiom, not the amber ⚠ warning
        # idiom (which means "succeeded / still proceeding") it used to borrow.
        return _err("the env was uploaded but the server returned no id")
    if render.styled():
        print(render.env_published(slug))
    else:
        print(f"published {slug}")
        print(f'reference it in your config:\n\n  [environment]\n  id = "{slug}"')
    return 0


def cmd_env_push(args) -> int:
    import tempfile

    env_name = _normalize_env_name(str(getattr(args, "name", "") or ""))
    if not env_name:
        return _err("env name required: pass `--name <name>`")

    src = Path(args.path)
    if not src.exists():
        return _err(f"no such path: {src}")

    if src.is_dir():
        canonical_entrypoint = src / _ENV_ENTRYPOINT
        if canonical_entrypoint.is_file():
            entrypoint = canonical_entrypoint
            env_root = src
        elif (src / "pyproject.toml").is_file():
            return _err(f"{src} has a pyproject.toml but no environment.py entrypoint")
        else:
            modules = [p for p in sorted(src.glob("*.py")) if not p.name.startswith("__")]
            if len(modules) != 1:
                return _err(
                    f"{src} has no environment.py and "
                    f"{'no' if not modules else 'multiple'} top-level .py module(s); "
                    "add an environment.py entrypoint or pass the exact .py file "
                    "for a single-file smoke test."
                )
            env_root = src
            entrypoint = modules[0]
    elif src.is_file() and src.suffix == ".py":
        env_root = src.parent
        entrypoint = src
    else:
        return _err(f"cannot publish {src}: expected a Freesolo .py module or an env directory.")

    with tempfile.TemporaryDirectory(prefix="flash-env-push-") as tmp:
        pkg = Path(tmp)
        module_source = entrypoint.read_text()
        (pkg / _ENV_ENTRYPOINT).write_text(_with_syspath_bootstrap(module_source))
        _copy_env_sidecars(env_root, pkg, entrypoint=entrypoint)
        # Only synthesize a stub README when the env didn't ship its own (now carried as a
        # ``.md`` sidecar) — don't clobber a user-authored README with boilerplate.
        readme = pkg / "README.md"
        if not readme.exists():
            readme.write_text(f"# {env_name}\n\nFlash Freesolo environment.\n")
        # One progress widget spans both phases the user otherwise waits through silently:
        # packaging (walk + gzip, slow for large datasets) and the upload itself.
        bar = _UploadProgress(env_name)
        bar.status("packaging environment")
        try:
            package_b64 = _tar_b64(pkg)
            return _upload_and_report(env_name, package_b64=package_b64, bar=bar)
        finally:
            bar.clear()
