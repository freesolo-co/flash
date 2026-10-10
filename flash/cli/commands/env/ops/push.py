"""Environment publish/pull machinery for the `flash env` subcommands.

`flash env push` packages a local Freesolo environment and uploads it through the
managed Flash control plane.
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from flash._internal.channel import CLI_NAME
from flash.cli.ui import render
from flash.cli.ui.tty import TtyStatusLine
from flash.envs.package.limits import ARCHIVE_MEMBER_LIMIT as _ENV_PUSH_MAX_FILES

if TYPE_CHECKING:
    from flash.client.http import ProgressCallback


# `list2cmdline` quotes argv for the C runtime, not cmd.exe or powershell. reject metacharacters
# from both shells because pasted suggestions execute them before the program sees the argument.
_CMD_METACHARACTERS = frozenset('&|<>^()!"%' + ";{}[]$`,'@#")


def _on_windows() -> bool:
    """Return whether shell suggestions need Windows quoting.

    This seam lets tests simulate Windows without mutating process-wide ``os.name``, which breaks
    ``pathlib.Path`` on Python 3.11 CI.
    """
    return os.name == "nt"


def _quote_shell_token(token: str) -> str:
    """Quote one token for the platform shell, or return empty when unsafe.

    POSIX quoting does not group cmd.exe arguments. Unsafe Windows metacharacters suppress the
    copy-pasteable form rather than risk executing another command.
    """
    if not _on_windows():
        return shlex.quote(token)
    import subprocess

    # a destination named `foo&bar` comes back from list2cmdline unquoted, and pasting that hint
    # runs `bar` as a second command. there is no quoting that both survives cmd.exe and reaches
    # the C runtime intact for every one of these, so refuse to suggest a command at all rather
    # than print one that executes unintended text.
    if _CMD_METACHARACTERS & set(token):
        return ""
    return subprocess.list2cmdline([token])


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
    from flash.envs.loading.adapter import is_managed_environment_slug

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

    Environments are addressed by their managed hub slug ``namespace/project/name`` and pulled as package
    tarballs through the authenticated Flash control plane.
    """
    from flash.client import ClientError, client_from_config
    from flash.envs.loading.pull import (
        download_environment_file_from_archive,
        ensure_environment_pull_destination_available,
        environment_local_dirname,
        pull_environment_package_from_archive,
    )

    env_id = str(args.env_id or "").strip()
    managed_env_id, managed_error = _normalize_managed_hub_id(env_id)
    if managed_error == "not-canonical":
        return _err(
            f'env id must be lowercase "namespace/project/name" with no spaces (got {args.env_id!r})'
        )
    if managed_env_id is None:
        print(
            'env id must be a managed Freesolo hub slug "your-org/your-project/your-env" '
            f"(got {args.env_id!r})",
            file=sys.stderr,
        )
        return 1
    env_id = managed_env_id
    try:
        if args.path:
            positional = Path(args.path.replace("\\", "/"))
            default_name = positional.name
            out = Path(args.output) if args.output else Path(default_name)
            # the second positional names a path inside the env, not the destination. detect only
            # destination-shaped directories when -o is absent, using the original positional so
            # absolute paths and `..` survive basename reduction. multi-component relative paths
            # remain valid in-env paths even when a same-named local directory exists.
            traverses_up = ".." in positional.parts
            could_be_destination = positional == out or positional.is_absolute() or traverses_up
            mistaken_dest = (
                positional
                if (
                    not args.output
                    and could_be_destination
                    and positional.is_dir()
                    and not positional.is_symlink()
                )
                else None
            )
            if mistaken_dest is not None:
                # attached `--output=X`, not `-o X`: a directory whose basename starts with a
                # dash (./-dest -> -dest) is read by argparse as an option and rejected with
                # "expected one argument". quoting alone does not help -- the attached form
                # does, because the value cannot start the token. quote the whole token so it
                # also survives spaces and metacharacters on the way back through a shell.
                quoted = _quote_shell_token(f"--output={mistaken_dest}")
                # ...and when even that cannot be made safe (cmd.exe metacharacters), name the
                # option without the value rather than print a command that would run something
                # else when pasted.
                suggestion = quoted or "--output=DEST"
                # the whole-environment path refuses any nonempty destination, so recommending
                # it verbatim into a populated directory just trades this error for the next.
                if any(mistaken_dest.iterdir()):
                    # ...and --force is not the answer either when the destination contains the
                    # cwd: ensure_environment_pull_destination_available() rejects that outright
                    # rather than deleting the directory the user is standing in. ask the guard
                    # itself so this cannot drift from the rule it is describing.
                    from flash.envs.loading.pull import _cwd_is_inside

                    if _cwd_is_inside(mistaken_dest):
                        hint = (
                            f"\nhint: {mistaken_dest}{os.sep} is a directory, and the second "
                            f"positional is a path INSIDE the environment, not a destination. to "
                            f"download the whole environment, pass a destination with -o -- but not "
                            f"{mistaken_dest}{os.sep}, which contains the current working directory "
                            f"and cannot be replaced even with --force. choose a separate path: "
                            f"{CLI_NAME} env pull {env_id} --output=DEST"
                        )
                    else:
                        hint = (
                            f"\nhint: {mistaken_dest}{os.sep} is a directory, and the second "
                            f"positional is a path INSIDE the environment, not a destination. to "
                            f"download the whole environment, pass a destination with -o; "
                            f"{mistaken_dest}{os.sep} is not empty, so choose a new path or replace "
                            f"its contents: {CLI_NAME} env pull {env_id} {suggestion} --force"
                        )
                else:
                    hint = (
                        f"\nhint: to download the whole environment into {mistaken_dest}{os.sep}, "
                        f"drop the second positional and pass it as the destination: "
                        f"{CLI_NAME} env pull {env_id} {suggestion}"
                    )
                print(
                    f"refusing to overwrite directory {mistaken_dest} with a file "
                    "(a single-file pull needs -o to be a FILE path, not a directory)" + hint,
                    file=sys.stderr,
                )
                return 1
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
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"env pull failed: {exc}", file=sys.stderr)
        return 1


def cmd_env_delete(args) -> int:
    from flash.client import ClientError, client_from_config
    from flash.core.spec import require_project_id

    try:
        project_id = require_project_id(getattr(args, "project", None))
    except (TypeError, ValueError) as exc:
        detail = str(exc).replace("project", "project id", 1)
        return _err(f"{detail}: pass `--project <project-uuid>`")

    # Delete only targets MANAGED hub ids ("namespace/project/name") — not github: refs or local paths, which
    # don't live on the hub. And enforce the hub's canonical form (lowercase, no surrounding
    # whitespace) BEFORE the network call: the server's slug validator is lowercase-only, so a
    # mixed-case / padded id would otherwise make a pointless request and return a confusing 400.
    env_id, validation_error = _normalize_managed_hub_id(args.env_id)
    if validation_error == "not-managed":
        return _err(
            f'env id must be a managed Freesolo hub id "namespace/project/name" (got {args.env_id!r}); '
            "github refs and local paths can't be deleted from the hub"
        )
    if validation_error == "not-canonical" or env_id is None:
        return _err(
            f'env id must be lowercase "namespace/project/name" with no spaces (got {args.env_id!r})'
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
        result = client_from_config().delete_env(env_id, project_id=project_id)
    except ClientError as exc:
        return _err(str(exc))
    deleted = bool(result["deleted"])
    msg = f"deleted {env_id}" if deleted else f"{env_id} was not found on the hub (already deleted)"
    print(render.ok(msg) if render.styled() else msg)
    return 0


_ENV_ENTRYPOINT = "environment.py"
# the held-out evaluation sidecar (flash/envs/meta/evaluations.py). a known filename beside the
# entrypoint rather than an environment module: never an entrypoint candidate, and carried
# alongside the entrypoint on a single-file push the way its docs and dataset are.
_ENV_EVALUATIONS_SIDECAR = "evaluations.py"
_ENV_SYSPATH_BOOTSTRAP = (
    "import os as _flash_os, sys as _flash_sys\n"
    "_flash_sys.path.insert(0, _flash_os.path.dirname(__file__))\n"
)
_ENV_PUSH_IGNORED_NAMES = frozenset(
    {
        ".prime",
        ".git",
        ".github",
        "__pycache__",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        ".ds_store",
        ".git-worktrees",
        ".env",
        ".netrc",
        ".aws",
        ".ssh",
        "venv",
        "node_modules",
        "credentials",
        "secrets",
    }
)
_ENV_PUSH_ROOT_IGNORED_NAMES = frozenset({"pyproject.toml", "source"})
_ENV_PUSH_SECRET_PATTERNS = (
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "credentials*",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
)
# secret patterns match secret data FILES only (see _ignore_env_push_path): they never prune a
# directory, so a package tree like credentials_store/ still ships, and python source is exempt,
# so helper modules like credentials.py or id_ed25519_loader.py travel instead of being silently
# dropped and breaking the worker with ModuleNotFoundError. that keeps the prefixes broad enough
# to catch backup variants (credentials_backup, id_rsa_backup) without eating code or packages.
_ENV_PUSH_CODE_SUFFIXES = frozenset({".py", ".pyi"})
_ENV_PUSH_MAX_TOTAL_BYTES = 256 * 1024 * 1024


def _normalize_env_name(raw: str) -> str | None:
    """Normalize only the NAME segment; a ``<namespace>/<project>`` prefix passes through verbatim.

    The server is the sole authority on namespace and project grammar and ownership — rewriting
    either client-side would silently target a different (possibly forbidden) slug.
    """
    from flash.schema import normalize_env_name_segment

    text = str(raw or "").strip()
    if "/" not in text:
        return normalize_env_name_segment(text)
    parts = [part.strip() for part in text.split("/")]
    if len(parts) != 3 or not all(parts):
        return None
    name = normalize_env_name_segment(parts[2])
    if name is None:
        return None
    return f"{parts[0]}/{parts[1]}/{name}"


def _env_name_error(raw: str) -> str:
    """Why this `--name` was rejected, in the caller's own terms.

    A qualified id that is not three segments is the single most likely mistake right now: it is
    the form every script used before names became unique per project. Answering it with "env
    name required" sends the user looking for a missing flag they actually passed.
    """
    text = str(raw or "").strip()
    if not text:
        return "env name required: pass `--name <name>`"
    parts = [part.strip() for part in text.split("/")]
    # a correctly-shaped id that still failed normalized on its NAME segment, not its shape.
    # Blaming the missing project here would tell the user to add one they already passed.
    if len(parts) == 3 and all(parts):
        return f"env name invalid: {parts[2]!r} has no usable characters"
    if "/" in text:
        return (
            f"env name invalid: {text!r} is not `<namespace>/<project>/<name>`. "
            "environment names are unique per project, so a qualified id needs the project "
            "segment; or pass a bare `--name <name>` with `--project <project-uuid>`"
        )
    return f"env name invalid: {text!r} has no usable characters"


def _validate_env_entrypoint_source(source: bytes) -> None:
    """Reject exact Python bytes that cannot execute as an environment module."""
    compile(source, _ENV_ENTRYPOINT, "exec")


def _decode_env_entrypoint_source(source: bytes) -> tuple[str, str]:
    """Decode Python bytes with the interpreter's encoding-cookie and BOM contract."""
    import io
    import tokenize

    encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
    return source.decode(encoding), encoding


def _python_preamble_lines(lines: list[str]) -> int:
    """Return the shebang and encoding-cookie lines that must remain first."""
    if not lines:
        return 0
    encoding_cookie = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+", re.ASCII)
    blank_or_comment = re.compile(r"^[ \t\f]*(?:#|[\r\n]|$)", re.ASCII)
    if encoding_cookie.match(lines[0]):
        return 1
    if len(lines) > 1 and blank_or_comment.match(lines[0]) and encoding_cookie.match(lines[1]):
        return 2
    return 1 if lines[0].startswith("#!") else 0


def _with_syspath_bootstrap(env_source: str) -> str:
    """Insert the sys.path bootstrap after the Python module preamble."""
    import ast

    tree = ast.parse(env_source, filename=_ENV_ENTRYPOINT)
    lines = env_source.splitlines(keepends=True)
    insert_after = _python_preamble_lines(lines)
    body = tree.body
    i = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        insert_after = max(insert_after, body[0].end_lineno or 0)
        i = 1
    while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__":
        insert_after = max(insert_after, body[i].end_lineno or 0)
        i += 1
    prefix = "".join(lines[:insert_after])
    if prefix and not prefix.endswith(("\n", "\r")):
        prefix += "\n"
    published_source = prefix + _ENV_SYSPATH_BOOTSTRAP + "".join(lines[insert_after:])
    if not env_source.endswith(("\n", "\r")):
        published_source = published_source.removesuffix("\n")
    return published_source


def _prepare_env_entrypoint_source(env_source: bytes) -> bytes:
    """Validate the authored bytes and return the exact validated archive bytes."""
    _validate_env_entrypoint_source(env_source)
    decoded_source, encoding = _decode_env_entrypoint_source(env_source)
    published_source = _with_syspath_bootstrap(decoded_source).encode(encoding)
    _validate_env_entrypoint_source(published_source)
    return published_source


def _raise_walk_error(error: OSError) -> None:
    """fail env packaging loudly instead of silently skipping an unreadable directory."""
    raise error


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
            dirs[:] = sorted(d for d in dirs if d.lower() not in _ENV_PUSH_IGNORED_NAMES)
            for name in sorted(files):
                child = root_path / name
                tar.add(child, arcname=str(child.relative_to(directory)), recursive=False)
    return base64.b64encode(buf.getvalue()).decode()


def _ignore_env_push_path(path: Path, *, env_root: Path, entrypoint: Path) -> bool:
    """Return whether a source path must not be included in an env package."""
    import fnmatch

    name = path.name
    lowered_name = name.lower()
    if path.is_symlink():
        return True
    if path.is_dir() and (path / "pyvenv.cfg").is_file():
        return True
    if path == entrypoint or (path.parent == env_root and lowered_name == _ENV_ENTRYPOINT.lower()):
        return True
    if lowered_name.startswith(".") or lowered_name in _ENV_PUSH_IGNORED_NAMES:
        return True
    if path.parent == env_root and lowered_name in _ENV_PUSH_ROOT_IGNORED_NAMES:
        return True
    if (
        path.is_file()
        and path.suffix.lower() not in _ENV_PUSH_CODE_SUFFIXES
        and any(fnmatch.fnmatchcase(lowered_name, pattern) for pattern in _ENV_PUSH_SECRET_PATTERNS)
    ):
        return True
    return not (path.is_dir() or path.is_file())


def _iter_env_sidecar_files(env_root, *, entrypoint, include_full_tree):
    """Publishable files beside the entrypoint. Implemented in `.imports`.

    Kept as a wrapper rather than a re-exported name so the two call sites below still go
    through this module's attribute: tests patch `push._iter_env_sidecar_files` to count walk
    yields, and a `from ... import` binding would be resolved before that patch lands.
    """
    from flash.cli.commands.env.ops.imports import _iter_env_sidecar_files as _impl

    yield from _impl(env_root, entrypoint=entrypoint, include_full_tree=include_full_tree)


def _entrypoint_alias_source(module_name: str) -> str:
    """The alias module that rebinds a noncanonical entrypoint name to environment.py."""
    return (
        "# generated by `flash env push`: this environment's entrypoint was published as\n"
        f"# {_ENV_ENTRYPOINT}, so `import {module_name}` keeps resolving to the same module.\n"
        "import sys\n"
        "\n"
        "import environment as _environment\n"
        "\n"
        "sys.modules[__name__] = _environment\n"
    )


def _write_entrypoint_alias(pkg: Path, *, entrypoint: Path) -> None:
    """Keep a renamed entrypoint importable under its local module name.
    Rebind ``sys.modules`` so sidecars and the runner share state. The entrypoint is excluded from
    the sidecar walk, so the alias cannot collide with a packaged sibling.
    """
    if entrypoint.name == _ENV_ENTRYPOINT:
        return
    (pkg / entrypoint.name).write_text(_entrypoint_alias_source(entrypoint.stem))


def _check_env_push_limits(
    env_root: Path, *, entrypoint: Path, include_full_tree: bool, env_name: str
) -> None:
    """Reject oversized source trees before copying or building an in-memory archive."""
    files = 1
    total_bytes = entrypoint.stat().st_size + len(_ENV_SYSPATH_BOOTSTRAP.encode())
    if entrypoint.name != _ENV_ENTRYPOINT:
        # the import alias `cmd_env_push` writes beside environment.py counts against the same
        # member cap the server enforces, so it cannot be free here.
        files += 1
        total_bytes += len(_entrypoint_alias_source(entrypoint.stem).encode())
    directories: set[Path] = set()
    has_readme = False
    for child, relative in _iter_env_sidecar_files(
        env_root, entrypoint=entrypoint, include_full_tree=include_full_tree
    ):
        files += 1
        total_bytes += child.stat().st_size
        has_readme = has_readme or relative == Path("README.md")
        # the server counts every file AND directory against the same member cap when it
        # repackages the checkout for download, so count unique parent dirs here too.
        directories.update(parent for parent in relative.parents if parent != Path("."))
    if not has_readme:
        files += 1
        total_bytes += len(f"# {env_name}\n\nFlash Freesolo environment.\n".encode())

    members = files + len(directories)
    if total_bytes > _ENV_PUSH_MAX_TOTAL_BYTES:
        raise ValueError(
            f"environment package totals {_human_bytes(total_bytes)} "
            f"(limit {_human_bytes(_ENV_PUSH_MAX_TOTAL_BYTES)}); "
            "remove large artifacts or use a smaller dataset"
        )
    if members > _ENV_PUSH_MAX_FILES:
        raise ValueError(
            f"selected environment contains {members:,} files and directories "
            f"(limit {_ENV_PUSH_MAX_FILES:,}); "
            "remove large artifacts or use a smaller dataset"
        )


def _copy_env_sidecars(
    env_root: Path, dest: Path, *, entrypoint: Path, include_full_tree: bool
) -> None:
    """Copy publishable sidecar files beside environment.py, creating directories lazily."""
    import shutil

    for child, relative in _iter_env_sidecar_files(
        env_root, entrypoint=entrypoint, include_full_tree=include_full_tree
    ):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
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


def _upload_and_report(
    name: str, *, package_b64: str, bar: _UploadProgress, project_id: str
) -> int:
    """Upload a packaged env to the managed control plane and print the returned id."""
    from flash.client import ClientError, client_from_config

    try:
        result = client_from_config().publish_env(
            name=name, package_b64=package_b64, project_id=project_id, progress=bar.callback
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


def _resolve_local_env_entrypoint(path: str | Path) -> tuple[Path, Path, Path, bool]:
    """Resolve a local environment path to its Python entrypoint and package root."""
    src = Path(path)
    if not src.exists():
        raise ValueError(f"no such path: {src}")
    if src.is_symlink():
        raise ValueError(f"cannot publish {src}: symlinks are not allowed")

    include_full_tree = src.is_dir()
    if include_full_tree:
        canonical_entrypoint = src / _ENV_ENTRYPOINT
        if not canonical_entrypoint.is_file():
            # a directory package names its entrypoint `environment.py`. that is what `flash env
            # setup` scaffolds and the only layout the docs describe, so a directory without one is
            # rejected rather than guessed at: inferring the entrypoint from "the sole top-level
            # module" made the resolution depend on which OTHER files happened to be present, so
            # adding a second module turned a working push into a rejection.
            raise ValueError(
                f"{src} has no {_ENV_ENTRYPOINT} entrypoint; add one, "
                "or pass the exact .py file for a single-file smoke test."
            )
        entrypoint = canonical_entrypoint
        env_root = src
    elif src.is_file() and src.suffix == ".py":
        env_root = src.parent
        entrypoint = src
    else:
        raise ValueError(
            f"cannot publish {src}: expected a Freesolo .py module or an env directory."
        )

    if entrypoint.is_symlink():
        raise ValueError(f"cannot publish {entrypoint}: symlinks are not allowed")
    return src, env_root, entrypoint, include_full_tree


def cmd_env_push(args) -> int:
    raw_env_name = str(getattr(args, "name", "") or "")
    env_name = _normalize_env_name(raw_env_name)
    if not env_name:
        return _err(_env_name_error(raw_env_name))

    from flash.core.spec import require_project_id

    try:
        project_id = require_project_id(getattr(args, "project", None))
    except (TypeError, ValueError) as exc:
        detail = str(exc).replace("project", "project id", 1)
        return _err(f"{detail}: pass `--project <project-uuid>`")

    try:
        src, env_root, entrypoint, include_full_tree = _resolve_local_env_entrypoint(args.path)
    except ValueError as exc:
        return _err(str(exc))

    try:
        published_source = _prepare_env_entrypoint_source(entrypoint.read_bytes())
    except (SyntaxError, UnicodeError, ValueError) as exc:
        location = (
            f"line {exc.lineno}" if getattr(exc, "lineno", None) is not None else "unknown line"
        )
        return _err(
            f"cannot publish {src}: environment entrypoint has invalid Python syntax ({location})"
        )
    except OSError:
        return _err(f"cannot publish {src}: environment entrypoint could not be read")

    try:
        _check_env_push_limits(
            env_root,
            entrypoint=entrypoint,
            include_full_tree=include_full_tree,
            env_name=env_name,
        )
    except (OSError, ValueError) as exc:
        return _err(f"cannot publish {src}: {exc}")

    with tempfile.TemporaryDirectory(prefix="flash-env-push-") as tmp:
        pkg = Path(tmp)
        (pkg / _ENV_ENTRYPOINT).write_bytes(published_source)
        _write_entrypoint_alias(pkg, entrypoint=entrypoint)
        _copy_env_sidecars(
            env_root,
            pkg,
            entrypoint=entrypoint,
            include_full_tree=include_full_tree,
        )
        # Only synthesize a stub README when the env didn't ship its own (now carried as a
        # ``.md`` sidecar) — don't clobber a user-authored README with boilerplate.
        readme = pkg / "README.md"
        if not readme.exists():
            readme.write_text(f"# {env_name}\n\nFlash Freesolo environment.\n")
        from flash.envs.package.direct_tokens import (
            DirectTokenScanError,
            package_contains_direct_token,
        )

        try:
            contains_direct_token = package_contains_direct_token(pkg)
        except DirectTokenScanError:
            return _err("environment package could not be scanned safely")
        if contains_direct_token:
            return _err(
                "environment package contains a direct access token; remove it before publishing"
            )
        # one progress widget spans both phases the user otherwise waits through silently:
        # packaging (walk + gzip, slow for large datasets) and the upload itself.
        bar = _UploadProgress(env_name)
        bar.status("packaging environment")
        try:
            package_b64 = _tar_b64(pkg)
            return _upload_and_report(
                env_name, package_b64=package_b64, bar=bar, project_id=project_id
            )
        finally:
            bar.clear()
